#!/usr/bin/env python3
"""
rtenzo-gerar-dataset.py — gera datasets .jsonl (Enzo-T) a partir das fanfics do rtenzo.

Schema (um registro por linha, nesta ordem):
    id          enzo-t-XXXX
    title       título da história
    story       história completa
    tags        lista de tags            (gerada por IA)
    characters  lista de personagens     (gerada por IA)
    places      lista de lugares         (gerada por IA)
    universe    universo da fanfic       (pasta)
    words       quantidade de palavras
    writer      autor da fanfic          (pasta)

ESTRUTURA DO ZIP / PASTA
    rtenzo-fanfics/
        universe/
            writer/
                story.txt            (um arquivo = uma história)
                story/               (ou uma pasta com vários .txt/.md = capítulos,
                    01.txt            que são juntados na ordem do nome)
                    02.txt

MODELO BASE PARA tags / characters / places
    darrellbest/Qwen3.5-2B-Heretic  (baixado do Hugging Face, roda local)
    Precisa de: pip install transformers torch accelerate
    Sem GPU/modelo: use --no-llm (tags/characters/places saem como listas vazias).

USO
    python rtenzo-gerar-dataset.py                       # pede o .zip no prompt
    python rtenzo-gerar-dataset.py rtenzo-fanfics.zip    # gera rtenzo-fanfics.jsonl
    python rtenzo-gerar-dataset.py rtenzo-fanfics.zip --no-llm --prefix enzo-t2

Front matter opcional no arquivo (--- chave: valor ---): title, universe, writer.
Fora o modelo de IA, só usa a biblioteca padrão do Python (3.8+).
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

FIELDS = [
    "id", "title", "story", "tags", "characters", "places",
    "universe", "words", "writer",
]
FRONT_MATTER_KEYS = {"title", "universe", "writer"}
EXTENSIONS = {".txt", ".md"}
WRAPPER_NAME = "rtenzo-fanfics"
DEFAULT_MODEL = "darrellbest/Qwen3.5-2B-Heretic"


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
        candidates.append(re.sub(r"\\(.)", r"\1", unquoted))
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
    """Tira a pasta-embrulho (rtenzo-fanfics ou com o nome do zip), se houver."""
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        if slugify(entries[0].name) in {slugify(WRAPPER_NAME), slugify(zip_stem)}:
            return entries[0]
    return root


# --------------------------------------------------------------------------- #
# Leitura e limpeza
# --------------------------------------------------------------------------- #
def parse_front_matter(raw: str) -> tuple[dict, str]:
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


def pretty_title(stem: str) -> str:
    """'my_story-name' -> 'my story name' (só troca _ por espaço quando não há espaços)."""
    return stem.replace("_", " ").strip() if " " not in stem else stem.strip()


def text_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


# --------------------------------------------------------------------------- #
# Descoberta: cada "história" é um arquivo ou uma pasta de capítulos
# --------------------------------------------------------------------------- #
def is_text(p: Path) -> bool:
    return (p.is_file() and p.suffix.lower() in EXTENSIONS
            and not p.name.startswith(".") and p.stem.lower() != "readme")


def discover(root: Path) -> list[tuple[Path, list[Path], tuple[str, ...]]]:
    """Devolve (caminho da história, arquivos em ordem, pastas acima dela)."""
    if root.is_file():
        return [(root, [root], ())]
    stories = []
    for p in sorted(root.rglob("*"), key=natural_key):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        folders = rel.parts[:-1]
        if is_text(p):
            # arquivo dentro de uma pasta de história (writer/story/cap.txt)?
            if len(folders) >= 3:
                continue  # entra pela pasta da história, mais abaixo
            stories.append((p, [p], folders))
        elif p.is_dir() and len(rel.parts) == 3:  # universe/writer/story/
            files = sorted((f for f in p.rglob("*") if is_text(f)
                            and not any(x.startswith(".") for x in f.relative_to(p).parts)),
                           key=lambda f: [natural_key(Path(x)) for x in f.relative_to(p).parts])
            if files:
                stories.append((p, files, rel.parts[:-1]))
    return stories


# --------------------------------------------------------------------------- #
# IA: tags, characters, places
# --------------------------------------------------------------------------- #
PROMPT = (
    "Read the story below and extract metadata. Reply with ONLY a JSON object with "
    'exactly these keys: "tags" (5 to 15 short lowercase descriptive tags: genres, '
    'themes, tropes, kinks), "characters" (names of the characters that appear), '
    '"places" (names of the places/locations where it happens). Use [] when there is '
    "nothing. Keep the story's original language for names.\n\nSTORY:\n{story}"
)


class Tagger:
    """Carrega o modelo uma vez e extrai tags/characters/places de cada história."""

    def __init__(self, model_id: str, max_chars: int, max_new_tokens: int):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError:
            sys.exit("Faltam dependências da IA: pip install transformers torch accelerate "
                     "(ou use --no-llm).")
        import transformers

        print(f"🤖 carregando {model_id} …")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        last_exc: Exception | None = None
        self.model = None
        for cls_name in ("AutoModelForCausalLM", "AutoModelForMultimodalLM",
                         "AutoModelForImageTextToText"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(model_id, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if self.model is None:
            sys.exit(f"Não consegui carregar {model_id}: {last_exc}")
        self.max_chars = max_chars
        self.max_new_tokens = max_new_tokens

    def _sample(self, story: str) -> str:
        """Histórias longas: começo + meio + fim, para caber no contexto."""
        if len(story) <= self.max_chars:
            return story
        third = self.max_chars // 3
        mid = len(story) // 2
        return (story[:third] + "\n[...]\n" + story[mid - third // 2: mid + third // 2]
                + "\n[...]\n" + story[-third:])

    def extract(self, story: str) -> dict:
        messages = [{"role": "user", "content": PROMPT.format(story=self._sample(story))}]
        try:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        reply = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_metadata(reply)


def _clean_list(value, limit: int = 30) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,;\n]", value)
    if not isinstance(value, list):
        return []
    seen, result = set(), []
    for item in value:
        item = re.sub(r"\s+", " ", str(item)).strip(" -*\"'")
        if item and item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result[:limit]


def parse_metadata(reply: str) -> dict:
    """Pega o 1º objeto JSON da resposta do modelo (ignora texto/cercas ao redor)."""
    data: dict = {}
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    start = reply.find("{")
    while start != -1 and not data:
        depth = 0
        for i in range(start, len(reply)):
            depth += {"{": 1, "}": -1}.get(reply[i], 0)
            if depth == 0:
                try:
                    data = json.loads(reply[start:i + 1])
                except json.JSONDecodeError:
                    pass
                break
        start = reply.find("{", start + 1)
    return {k: _clean_list(data.get(k)) for k in ("tags", "characters", "places")}


# --------------------------------------------------------------------------- #
# Construção dos registros
# --------------------------------------------------------------------------- #
def build_record(path: Path, files: list[Path], folders: tuple[str, ...],
                 root: Path, defaults: dict, flatten: bool) -> dict:
    front: dict = {}
    parts = []
    for i, f in enumerate(files):
        fm, body = parse_front_matter(f.read_text(encoding="utf-8", errors="replace"))
        if i == 0:
            front = fm
        body = clean_body(body, flatten)
        if body:
            parts.append(body)
    story = (" " if flatten else "\n\n").join(parts)
    if not story:
        raise ValueError("arquivo vazio")

    # universe/writer vêm das duas primeiras pastas (a 1ª é o universo, a 2ª o autor)
    from_folders: dict = {}
    if len(folders) >= 1:
        from_folders["universe"] = folders[0]
    if len(folders) >= 2:
        from_folders["writer"] = folders[1]
    info = {**defaults, **from_folders, **front}

    title = info.get("title") or pretty_title(path.stem if path.is_file() else path.name)
    universe = info.get("universe")
    return {
        "id": None,
        "title": title,
        "story": story,
        "tags": [],
        "characters": [],
        "places": [],
        "universe": slugify(universe) if universe else None,
        "words": len(story.split()),
        "writer": info.get("writer"),
        "_sort": ((universe or "").lower(), (info.get("writer") or "").lower(),
                  title.lower(), str(path)),
        "_fp": text_fingerprint(story),
    }


def generate(root: Path, out: Path, defaults: dict, args: argparse.Namespace) -> None:
    stories = discover(root)
    if not stories:
        sys.exit("Nenhuma história (.txt/.md) encontrada.")

    base = root if root.is_dir() else root.parent

    def shown(p: Path) -> str:
        try:
            return str(p.relative_to(base))
        except ValueError:
            return str(p)

    records, skipped = [], []
    seen: set[str] = set()
    for path, files, folders in stories:
        try:
            rec = build_record(path, files, folders, root, defaults, args.flatten)
        except Exception as exc:  # noqa: BLE001
            skipped.append((shown(path), str(exc)))
            continue
        if rec["words"] < args.min_words:
            skipped.append((shown(path), f"abaixo do mínimo ({rec['words']} palavras)"))
            continue
        if not args.no_dedupe:
            if rec["_fp"] in seen:
                skipped.append((shown(path), "duplicado"))
                continue
            seen.add(rec["_fp"])
        records.append(rec)

    records.sort(key=lambda r: r["_sort"])

    if records and not args.no_llm:
        tagger = Tagger(args.model, args.max_chars, args.max_new_tokens)
        for n, rec in enumerate(records, 1):
            meta = tagger.extract(rec["story"])
            rec.update(meta)
            print(f"  [{n}/{len(records)}] {rec['title']}: "
                  f"{len(meta['tags'])} tags, {len(meta['characters'])} personagens, "
                  f"{len(meta['places'])} lugares")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, rec in enumerate(records, start=args.start):
            rec["id"] = f"{args.prefix}-{i:0{args.width}d}"
            row = {k: rec[k] for k in FIELDS}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_words = sum(r["words"] for r in records)
    print(f"✔ {len(records)} registros -> {out.resolve()}")
    if records:
        print(f"  ids: {records[0]['id']} … {records[-1]['id']}")
        print(f"  autores: {len({r['writer'] for r in records})}")
    print(f"  {total_words:,} palavras")
    counts = Counter(r["universe"] for r in records)
    if counts:
        print("  universe: " + ", ".join(f"{k}={v}" for k, v in counts.most_common()))
    if skipped:
        print(f"⚠ {len(skipped)} item(ns) ignorado(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Gera JSONL Enzo-T a partir das fanfics do rtenzo (.zip, pasta ou arquivo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("input", nargs="?",
                    help=".zip, pasta ou .txt/.md (se omitido, o script pergunta)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="saída .jsonl (padrão: <nome-do-zip>.jsonl na pasta atual)")
    ap.add_argument("--prefix", default="enzo-t", help="prefixo do id (padrão: enzo-t)")
    ap.add_argument("--start", type=int, default=1, help="número do primeiro id (padrão: 1)")
    ap.add_argument("--width", type=int, default=4, help="zeros à esquerda no id (padrão: 4)")
    ap.add_argument("--universe", help="universe padrão quando não há pasta de universo")
    ap.add_argument("--writer", help="writer padrão quando não há pasta de autor")
    ap.add_argument("--flatten", action="store_true",
                    help="colapsa quebras de linha/espaços da história em espaços simples")
    ap.add_argument("--min-words", type=int, default=0, help="descarta histórias menores")
    ap.add_argument("--no-dedupe", action="store_true", help="não remove histórias duplicadas")
    ap.add_argument("--no-llm", action="store_true",
                    help="não usa IA: tags/characters/places saem vazios")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"padrão: {DEFAULT_MODEL}")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="máx. de caracteres da história enviados ao modelo (padrão: 12000)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    raw_in = args.input
    if not raw_in:
        try:
            raw_in = input("Cole ou arraste o .zip (ou pasta) aqui e aperte Enter: ")
        except EOFError:
            sys.exit("Nenhuma entrada informada.")
    src = resolve_input(raw_in)

    defaults = {k: v for k, v in {"universe": args.universe, "writer": args.writer}.items()
                if v is not None}

    tmp = None
    try:
        if src.is_file() and zipfile.is_zipfile(src):
            tmp = tempfile.TemporaryDirectory(prefix="rtenzo_dataset_")
            n = extract_zip(src, Path(tmp.name))
            if n == 0:
                sys.exit("O zip não tem nenhum .txt/.md.")
            root = unwrap(Path(tmp.name), src.stem)
            print(f"📦 {src.name}: {n} arquivo(s) de texto")
            default_name = src.stem
        else:
            root = unwrap(src, src.stem) if src.is_dir() else src
            default_name = src.stem if src.is_file() else src.resolve().name
        out = args.out or Path.cwd() / f"{default_name}.jsonl"
        generate(root, out, defaults, args)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
