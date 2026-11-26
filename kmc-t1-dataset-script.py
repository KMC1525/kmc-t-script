#!/usr/bin/env python3
"""
kmc-t1-dataset-script.py — gera o dataset KMC-T1 (.jsonl) a partir de um .zip de obras.

USO (arrastar e soltar)
    python kmc-t1-dataset-script.py
    -> arraste o .zip para a janela do terminal e aperte Enter: o dataset é gerado
       na mesma pasta do zip (<nome-do-zip>.jsonl).

    python kmc-t1-dataset-script.py kmc-t.zip              # direto, sem perguntar
    python kmc-t1-dataset-script.py kmc-t.zip -o saida.jsonl --prefix kmc-t1

ESTRUTURA DO ZIP
    kmc-t/
        type/                      erotica | fanfics
            sub-type/              naruto, fat, ...
                language/          ptbr | eng
                    title/         nome da obra
                        system.txt         (system prompt)
                        prompt1.txt        output1.txt
                        prompt2.txt        output2.txt
                        ...                (quantos turnos quiser)
    Os arquivos podem ficar direto na pasta da obra ou numa subpasta "messages/".
    O nome da pasta raiz (kmc-t) é opcional.

    Nomes aceitos (qualquer extensão .txt/.md, número em qualquer posição):
        system  : system, system_prompt, sistema
        prompt N: prompt1, prompt_2, 3-prompt, input1, user1, pergunta1
        output N: output1, output_2, 3-output, response1, assistant1, resposta1
    O turno N só entra se tiver prompt N e output N.

SCHEMA (um registro por linha, nesta ordem)
    id         kmc-t1-0001
    type       pasta type
    sub-type   pasta sub-type
    language   pasta language
    title      pasta da obra
    num-turns  quantidade de turnos (pares prompt + output)
    words      palavras de todas as mensagens somadas
    messages   [{"role": "system"|"user"|"assistant", "content": ...}, ...]

O zip é lido em memória (nada é extraído). Só usa a biblioteca padrão (Python 3.8+).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse

FIELDS = ["id", "type", "sub-type", "language", "title", "num-turns", "words", "messages"]
EXTENSIONS = {".txt", ".md"}
ROOT_NAME = "kmc-t"
MESSAGES_DIR = "messages"
SYSTEM_WORDS = ("system", "sistema")
PROMPT_WORDS = ("prompt", "input", "user", "pergunta", "entrada")
OUTPUT_WORDS = ("output", "response", "assistant", "resposta", "saida", "saída")


# --------------------------------------------------------------------------- #
# Entrada: caminho arrastado/colado
# --------------------------------------------------------------------------- #
def resolve_input(raw: str) -> Path:
    """Aceita caminho com aspas, espaços escapados (arrastar no terminal) ou file://."""
    s = raw.strip()
    if s.startswith("& "):  # PowerShell: & 'C:\pasta\arquivo.zip'
        s = s[2:].strip()
    if s.startswith("file://"):
        s = unquote(urlparse(s).path)
    unquoted = s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s
    candidates = [unquoted]
    if os.name != "nt":
        candidates.append(re.sub(r"\\(.)", r"\1", unquoted))  # 'meu\ arquivo.zip'
    for cand in candidates:
        p = Path(cand).expanduser()
        if p.is_file():
            return p
    sys.exit(f"Não encontrei o arquivo: {raw.strip()}")


# --------------------------------------------------------------------------- #
# Leitura do zip (em memória)
# --------------------------------------------------------------------------- #
def _member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not (info.flag_bits & 0x800):  # sem flag UTF-8 o Python assume cp437
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def decode_text(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    return text.strip()


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def locate_work(dirs: tuple[str, ...]) -> tuple[str, str, str, str] | str:
    """(type, sub-type, language, title) a partir das pastas, ou o motivo de não servir."""
    if dirs and dirs[-1].lower() == MESSAGES_DIR:
        dirs = dirs[:-1]
    lowered = [d.lower() for d in dirs]
    if ROOT_NAME in lowered:
        dirs = dirs[lowered.index(ROOT_NAME) + 1:]
        if len(dirs) != 4:
            return f"esperava type/sub-type/language/title, achei {len(dirs)} nível(is)"
        return tuple(dirs)  # type: ignore[return-value]
    if len(dirs) < 4:
        return f"pastas insuficientes ({len(dirs)}); precisa de type/sub-type/language/title"
    return tuple(dirs[-4:])  # type: ignore[return-value]


def read_zip(zip_path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Agrupa os arquivos de texto por obra: {(type, sub, lang, title): [(arquivo, texto)]}."""
    works: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    ignored: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = _member_name(info).replace("\\", "/")
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if (not parts or ".." in parts or "__MACOSX" in parts
                    or any(p.startswith(".") for p in parts)):
                continue
            if Path(parts[-1]).suffix.lower() not in EXTENSIONS:
                continue
            where = locate_work(tuple(parts[:-1]))
            if isinstance(where, str):
                ignored.append((name, where))
                continue
            works[where].append((parts[-1], decode_text(zf.read(info))))
    return works, ignored


# --------------------------------------------------------------------------- #
# Construção dos registros
# --------------------------------------------------------------------------- #
def classify(filename: str) -> tuple[str | None, int | None]:
    """('system'|'prompt'|'output'|None, número do turno ou None)."""
    stem = Path(filename).stem.lower()
    if any(w in stem for w in SYSTEM_WORDS):
        return "system", None
    num = re.search(r"\d+", stem)
    number = int(num.group()) if num else None
    if any(w in stem for w in PROMPT_WORDS):
        return "prompt", number
    if any(w in stem for w in OUTPUT_WORDS):
        return "output", number
    return None, None


def build_messages(files: list[tuple[str, str]]) -> tuple[list[dict], list[str]]:
    """Monta a lista de mensagens de uma obra. Devolve (messages, avisos)."""
    notes: list[str] = []
    system: str | None = None
    prompts: dict[int, str] = {}
    outputs: dict[int, str] = {}

    for filename, text in sorted(files, key=lambda f: natural_key(f[0])):
        role, number = classify(filename)
        if role is None:
            notes.append(f"{filename}: nome não reconhecido, ignorado")
        elif not text:
            notes.append(f"{filename}: vazio, ignorado")
        elif role == "system":
            if system is None:
                system = text
            else:
                notes.append(f"{filename}: system prompt repetido, ignorado")
        elif number is None:
            notes.append(f"{filename}: sem número do turno, ignorado")
        else:
            target = prompts if role == "prompt" else outputs
            if number in target:
                notes.append(f"{filename}: {role} {number} repetido, ignorado")
            else:
                target[number] = text

    for n in sorted(set(prompts) ^ set(outputs)):
        missing = "output" if n in prompts else "prompt"
        notes.append(f"turno {n}: falta o {missing}, turno ignorado")

    messages: list[dict] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for n in sorted(set(prompts) & set(outputs)):
        messages.append({"role": "user", "content": prompts[n]})
        messages.append({"role": "assistant", "content": outputs[n]})
    return messages, notes


def build_records(works: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    records: list[dict] = []
    notes: list[tuple[str, str]] = []
    for key in sorted(works, key=lambda k: [natural_key(x) for x in k]):
        type_, sub_type, language, title = key
        label = "/".join(key)
        messages, warns = build_messages(works[key])
        notes.extend((label, w) for w in warns)
        turns = sum(1 for m in messages if m["role"] == "user")
        if turns == 0:
            notes.append((label, "nenhum turno completo (prompt + output), obra ignorada"))
            continue
        if not messages or messages[0]["role"] != "system":
            notes.append((label, "sem system prompt"))
        records.append({
            "id": None,
            "type": type_,
            "sub-type": sub_type,
            "language": language,
            "title": title,
            "num-turns": turns,
            "words": sum(len(m["content"].split()) for m in messages),
            "messages": messages,
        })
    return records, notes


# --------------------------------------------------------------------------- #
# Saída
# --------------------------------------------------------------------------- #
def free_path(path: Path) -> Path:
    """Não sobrescreve: se já existe, usa nome-2.jsonl, nome-3.jsonl, ..."""
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.stem}-{n}{path.suffix}").exists():
        n += 1
    return path.with_name(f"{path.stem}-{n}{path.suffix}")


def generate(zip_path: Path, out: Path | None, prefix: str, start: int, width: int) -> int:
    print(f"📦 lendo {zip_path.name} …")
    try:
        works, ignored = read_zip(zip_path)
    except zipfile.BadZipFile:
        sys.exit("Esse arquivo não é um .zip válido.")
    if not works:
        sys.exit("Nenhuma obra encontrada (.txt/.md em type/sub-type/language/title).")

    records, notes = build_records(works)
    if not records:
        sys.exit("Nenhuma obra com turno completo (prompt + output).")

    out = out or free_path(zip_path.with_suffix(".jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, rec in enumerate(records, start=start):
            rec["id"] = f"{prefix}-{i:0{width}d}"
            fh.write(json.dumps({k: rec[k] for k in FIELDS}, ensure_ascii=False) + "\n")

    print(f"✔ {len(records)} obra(s) -> {out.resolve()}")
    print(f"  ids: {records[0]['id']} … {records[-1]['id']}")
    print(f"  {sum(r['num-turns'] for r in records):,} turnos, "
          f"{sum(r['words'] for r in records):,} palavras")
    for field in ("type", "sub-type", "language"):
        counts = Counter(r[field] for r in records)
        print(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    if ignored:
        print(f"⚠ {len(ignored)} arquivo(s) fora da estrutura:")
        for name, reason in ignored:
            print(f"  - {name}: {reason}")
    if notes:
        print(f"⚠ {len(notes)} aviso(s):")
        for label, note in notes:
            print(f"  - {label}: {note}")
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gera o dataset KMC-T1 (.jsonl) a partir de um .zip de obras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input", nargs="?", help=".zip das obras (se omitido, o script pergunta)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="saída .jsonl (padrão: <nome-do-zip>.jsonl ao lado do zip)")
    ap.add_argument("--prefix", default="kmc-t1", help="prefixo do id (padrão: kmc-t1)")
    ap.add_argument("--start", type=int, default=1, help="número do primeiro id (padrão: 1)")
    ap.add_argument("--width", type=int, default=4, help="zeros à esquerda no id (padrão: 4)")
    args = ap.parse_args()

    interactive = not args.input
    raw_in = args.input
    if interactive:
        print("KMC-T1 • gerador de dataset (.jsonl)")
        try:
            raw_in = input("Arraste o .zip para cá e aperte Enter: ")
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nNenhuma entrada informada.")
    try:
        generate(resolve_input(raw_in), args.out, args.prefix, args.start, args.width)
    finally:
        if interactive and os.name == "nt":  # janela aberta com duplo clique não fecha na hora
            try:
                input("\nEnter para sair…")
            except (EOFError, KeyboardInterrupt):
                pass


if __name__ == "__main__":
    main()
