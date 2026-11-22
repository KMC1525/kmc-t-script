#!/usr/bin/env python3
"""
gerar_dataset.py — gera datasets .jsonl no formato do KMCLabs/kmc-t-dataset.

Schema (um registro por linha, nesta ordem):
    id        kmc-t-XXXX
    title     título da obra
    chapter   número do capítulo da obra (int, ou null)
    story     história/capítulo completo
    source    fanfiction, literotica, rtenzo net etc.
    fandom    anime/manga, bimbo, naruto etc.
    words     quantidade de palavras
    language  idioma da história/capítulo

USO RÁPIDO
    python gerar_dataset.py                      # pede o .zip no prompt (cole ou arraste)
    python gerar_dataset.py fanfics.zip          # direto, gera fanfics.jsonl
    python gerar_dataset.py fanfics.zip --prefix kmc-t2 --start 1

A entrada pode ser um .zip, uma pasta ou um único .txt/.md. Dentro do zip,
cada .txt/.md é um texto (ou um capítulo, se o nome tiver "ch1", "chapter 2"...).

ESTRUTURA DE PASTAS (detectada sozinha, a menos que você passe --layout)
    source/fandom/obra/ch1.txt         ->  fanfiction / black-clover / Knight In Silver Armour
        (com 3 níveis de pasta, a pasta da obra manda no title e cada arquivo dentro
         dela é um capítulo, seja "ch1", "01", "Chapter 3 - Nome" ou "Prólogo";
         sem número no nome, o capítulo continua depois do maior número da obra)
    source/fandom/Obra - Ch1.txt       ->  fanfiction / black-clover / Obra
    source/Obra/ch1.txt                ->  só a source pela 1ª pasta; obra pela pasta
    source/Obra - Ch1.txt              ->  source pela pasta; obra pelo nome do arquivo
A pasta principal é sempre a source (fanfiction, ao3, rtenzo...) e as subpastas
dentro dela são os fandoms. Se o nome do arquivo já traz a obra, as pastas viram
source/fandom; se o arquivo é só "ch1"/"01", a última pasta é a obra.

METADADOS (menor -> maior prioridade)
    argumentos (--source, --fandom, --language; --source é só o padrão quando não há pasta)
    < estrutura de pastas < front matter no arquivo (--- chave: valor ---)

Só usa a biblioteca padrão do Python (3.8+).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

# Ordem exata das colunas do dataset original
FIELDS = [
    "id", "title", "chapter", "story", "source", "fandom", "words", "language",
]
FRONT_MATTER_KEYS = {"title", "story", "source", "language", "chapter", "fandom"}
LAYOUT_KEYS = {"source", "fandom", "title", "story"}
KEY_ALIASES = {"story": "title"}  # "story" nas pastas/front matter = nome da obra
SLUGGED_KEYS = {"source", "fandom"}  # "Black Clover" -> "black-clover"
EXTENSIONS = {".txt", ".md"}

# Detecta "ch1", "Ch01", "chapter 2", "cap_3", "capítulo-4" no nome do arquivo
CHAPTER_RE = re.compile(
    r"[\s_\-\.]*(?<![A-Za-z])(?:chapter|capitulo|capítulo|cap|ch)[\s_\-\.]*0*(\d+)(?!\d)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Entrada: caminho colado/arrastado, zip, pasta
# --------------------------------------------------------------------------- #
def resolve_input(raw: str) -> Path:
    """Aceita caminho com aspas, espaços escapados (arrastar no terminal) ou file://."""
    s = raw.strip()
    if s.startswith("file://"):
        s = unquote(urlparse(s).path)
    unquoted = s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s
    candidates = [unquoted]
    if os.name != "nt":
        candidates.append(re.sub(r"\\(.)", r"\1", unquoted))  # 'meu\ arquivo.zip'
    for cand in candidates:
        p = Path(cand).expanduser()
        if p.exists():
            return p
    sys.exit(f"Não encontrei: {raw.strip()}")


def _member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not (info.flag_bits & 0x800):  # sem flag UTF-8 o Python assume cp437
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def extract_zip(zip_path: Path, dest: Path) -> int:
    """Extrai só os .txt/.md, ignorando lixo (__MACOSX, ocultos) e caminhos perigosos."""
    dest = dest.resolve()
    count = 0
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
            target = dest.joinpath(*parts).resolve()
            if dest not in target.parents:  # zip-slip
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def slugify(value: str) -> str:
    return re.sub(r"[\W_]+", "-", value.lower(), flags=re.UNICODE).strip("-") or value


def unwrap(root: Path, zip_stem: str) -> Path:
    """Tira a pasta-embrulho que tem o mesmo nome do zip (padrão de Finder/Explorer)."""
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        if slugify(entries[0].name) == slugify(zip_stem):
            return entries[0]
    return root


def discover(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
        and p.stem.lower() != "readme"
    )


# --------------------------------------------------------------------------- #
# Leitura e limpeza
# --------------------------------------------------------------------------- #
def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Lê um bloco opcional '--- chave: valor ---' no topo do arquivo."""
    m = re.match(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?", raw, re.DOTALL)
    if not m:
        return {}, raw
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key not in FRONT_MATTER_KEYS:
            continue
        if key == "story":  # no schema novo "story" é o texto; aqui significa a obra
            key = "title"
        value = value.strip().strip("\"'")
        meta[key] = None if value.lower() in {"", "null", "none"} else value
    return meta, raw[m.end():]


def clean_body(body: str, flatten: bool) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    body = body.replace("﻿", "").strip()
    if flatten:
        return re.sub(r"\s+", " ", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def split_chapter(stem: str) -> tuple[str, int | None]:
    """'Knight In Silver Armour - Ch3' -> ('Knight In Silver Armour', 3); '07' -> ('', 7)."""
    if re.fullmatch(r"\s*\d+\s*", stem):
        return "", int(stem)
    m = CHAPTER_RE.search(stem)
    if not m:
        return stem.strip(" -_."), None
    base = (stem[: m.start()] + stem[m.end():]).strip(" -_.")
    return base, int(m.group(1))


def parse_layout(layout: str) -> list[str]:
    keys = [k.strip() for k in layout.replace("/", ",").split(",") if k.strip()]
    invalid = [k for k in keys if k not in LAYOUT_KEYS]
    if invalid:
        sys.exit(f"--layout inválido: {invalid}. Permitidos: {sorted(LAYOUT_KEYS)}")
    return [KEY_ALIASES.get(k, k) for k in keys]


def auto_meta(folders: tuple[str, ...], story_in_folder: bool) -> dict:
    """Infere source/fandom/obra só pela posição das pastas (1ª pasta = source)."""
    meta: dict = {}
    n = len(folders)
    if story_in_folder and n >= 1:
        meta["title"] = folders[-1]
        n -= 1
    outer = folders[:n]
    if len(outer) >= 1:
        meta["source"] = outer[0]
    if len(outer) >= 2:
        meta["fandom"] = outer[1]
    return meta


def text_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Construção dos registros
# --------------------------------------------------------------------------- #
def build_record(path: Path, root: Path, layout: list[str] | None,
                 defaults: dict, flatten: bool) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    front, body = parse_front_matter(raw)

    base = root.parent if root.is_file() else root
    folders = path.relative_to(base).parts[:-1]
    base_name, chapter_num = split_chapter(path.stem)

    # 3+ níveis (source/fandom/obra/): a pasta da obra define o title e o arquivo é só o capítulo
    chapter_file = layout is None and len(folders) >= 3
    if chapter_file:
        if chapter_num is None:  # nome sem "ch": usa o 1º número, ex. "03 - Nome.txt"
            m = re.search(r"\d+", path.stem)
            chapter_num = int(m.group()) if m else None
        base_name = ""

    if layout is None:
        from_folders = auto_meta(folders, story_in_folder=not base_name)
    else:
        from_folders = dict(zip(layout, folders))
    from_folders = {
        k: (slugify(v) if k in SLUGGED_KEYS else v) for k, v in from_folders.items()
    }

    info = {**defaults, **from_folders, **front}

    title = info.get("title") or base_name or (folders[-1] if folders else path.stem)
    auto_chapter = chapter_file and chapter_num is None and "chapter" not in front

    if "chapter" in front:
        digits = re.search(r"\d+", front["chapter"] or "")
        chapter = int(digits.group()) if digits else None
    else:
        chapter = chapter_num

    source = info.get("source")
    if not source:
        raise ValueError("sem 'source' (use --source, --layout ou front matter)")

    story = clean_body(body, flatten)
    if not story:
        raise ValueError("arquivo vazio")

    return {
        "id": None,  # preenchido depois, já com a ordem final
        "title": title,
        "chapter": chapter,
        "story": story,
        "source": source,
        "fandom": info.get("fandom"),
        "words": len(story.split()),
        "language": info.get("language") or "en",
        "_auto_chapter": auto_chapter,  # sem número no nome: numera pela ordem na pasta
        "_sort": (title.lower(), chapter if chapter is not None else 0, str(path)),
        "_fp": text_fingerprint(story),  # dedupe pelo texto
    }


def generate(root: Path, out: Path, layout: list[str] | None, defaults: dict,
             args: argparse.Namespace) -> None:
    files = discover(root)
    if not files:
        sys.exit("Nenhum .txt/.md encontrado.")

    def shown(p: Path) -> str:
        try:
            return str(p.relative_to(root if root.is_dir() else root.parent))
        except ValueError:
            return str(p)

    records, skipped = [], []
    seen: set[str] = set()
    for path in files:
        try:
            rec = build_record(path, root, layout, defaults, args.flatten)
        except Exception as exc:  # noqa: BLE001
            skipped.append((shown(path), str(exc)))
            continue
        if len(rec["story"]) < args.min_chars or rec["words"] < args.min_words:
            skipped.append((shown(path), f"abaixo do mínimo ({len(rec['story'])} chars, {rec['words']} palavras)"))
            continue
        if not args.no_dedupe:
            if rec["_fp"] in seen:
                skipped.append((shown(path), "duplicado"))
                continue
            seen.add(rec["_fp"])
        records.append(rec)

    # ordem final: por história e capítulo; ids contíguos depois dos filtros
    records.sort(key=lambda r: r["_sort"])
    # capítulos sem número no nome: continuam depois do maior número da obra (sem repetir)
    last: dict = {}
    for rec in records:
        key = (rec["source"], rec["fandom"], rec["title"])
        if not rec["_auto_chapter"] and rec["chapter"] is not None:
            last[key] = max(last.get(key, 0), rec["chapter"])
    for rec in records:
        if rec["_auto_chapter"]:
            key = (rec["source"], rec["fandom"], rec["title"])
            last[key] = last.get(key, 0) + 1
            rec["chapter"] = last[key]
            rec["_sort"] = (rec["_sort"][0], rec["chapter"], rec["_sort"][2])
    records.sort(key=lambda r: r["_sort"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, rec in enumerate(records, start=args.start):
            rec["id"] = f"{args.prefix}-{i:0{args.width}d}"
            row = {k: rec[k] for k in FIELDS}  # garante a ordem das colunas
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # resumo
    total_chars = sum(len(r["story"]) for r in records)
    total_words = sum(r["words"] for r in records)
    print(f"✔ {len(records)} registros -> {out.resolve()}")
    if records:
        print(f"  ids: {records[0]['id']} … {records[-1]['id']}")
        print(f"  obras: {len({r['title'] for r in records})}")
    print(f"  {total_chars:,} caracteres | {total_words:,} palavras")
    for field in ("source", "fandom", "language"):
        counts = Counter(r[field] for r in records)
        if counts:
            print(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    if skipped:
        print(f"⚠ {len(skipped)} arquivo(s) ignorado(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gera JSONL no formato do KMCLabs/kmc-t-dataset a partir de um .zip, pasta ou arquivo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input", nargs="?",
                    help=".zip, pasta ou .txt/.md (se omitido, o script pergunta)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="saída .jsonl (padrão: <nome-do-zip>.jsonl na pasta atual)")
    ap.add_argument("--prefix", default="kmc-t1", help="prefixo do id (padrão: kmc-t1)")
    ap.add_argument("--start", type=int, default=1, help="número do primeiro id (padrão: 1)")
    ap.add_argument("--width", type=int, default=4, help="zeros à esquerda no id (padrão: 4)")
    ap.add_argument("--source", default="fanfiction",
                    help="source padrão quando não há pasta de source (padrão: fanfiction)")
    ap.add_argument("--fandom", help="ex.: black-clover (padrão: pelas pastas)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--layout", default=None,
                    help="força o mapeamento das pastas, ex.: source/fandom/title "
                         "(padrão: detecta sozinho; '' = ignora pastas)")
    ap.add_argument("--flatten", action="store_true",
                    help="colapsa quebras de linha/espaços do texto em espaços simples")
    ap.add_argument("--min-chars", type=int, default=0, help="descarta textos menores")
    ap.add_argument("--min-words", type=int, default=0)
    ap.add_argument("--no-dedupe", action="store_true", help="não remove textos duplicados")
    args = ap.parse_args()

    raw_in = args.input
    if not raw_in:
        try:
            raw_in = input("Cole ou arraste o .zip (ou pasta) aqui e aperte Enter: ")
        except EOFError:
            sys.exit("Nenhuma entrada informada.")
    src = resolve_input(raw_in)

    layout = parse_layout(args.layout) if args.layout is not None else None
    defaults = {
        k: v for k, v in {
            "source": args.source, "fandom": args.fandom,
            "language": args.language,
        }.items() if v is not None
    }

    tmp = None
    try:
        if src.is_file() and zipfile.is_zipfile(src):
            tmp = tempfile.TemporaryDirectory(prefix="gerar_dataset_")
            n = extract_zip(src, Path(tmp.name))
            if n == 0:
                sys.exit("O zip não tem nenhum .txt/.md.")
            root = unwrap(Path(tmp.name), src.stem)
            print(f"📦 {src.name}: {n} arquivo(s) de texto")
            default_name = src.stem
        else:
            root = src
            default_name = src.stem if src.is_file() else src.resolve().name
        out = args.out or Path.cwd() / f"{default_name}.jsonl"
        generate(root, out, layout, defaults, args)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
