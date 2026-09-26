"""
vg_download_subset.py
=====================
Descarga las imágenes de un subset estratificado de Visual Genome.

Toma como entrada la salida de vg_stratified_subset.py y el fichero
image_data.json de VG para obtener las URLs de descarga.

Características:
  - Descarga paralela (configurable con --workers)
  - Resume automático: omite imágenes ya descargadas y válidas
  - Reintentos con backoff exponencial
  - Barra de progreso (tqdm si está instalado, fallback sin ella)
  - Log de fallos en un fichero separado para reinspección
  - Nombrado uniforme: <image_id>.jpg

Uso:
    python vg_download_subset.py \
        --subset     vg_subset_2000.json \
        --image_data /ruta/a/image_data.json \
        --output_dir ./vg_images \
        --workers    8 \
        --retries    3

Dependencias opcionales:
    pip install tqdm requests   (requests es más robusto que urllib para reintentos)
"""

import json
import time
import logging
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Intentar importar librerías opcionales
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Descarga de una imagen individual
# ---------------------------------------------------------------------------

def _download_with_requests(url: str, dest: Path, timeout: int) -> None:
    import requests as req
    resp = req.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def _download_with_urllib(url: str, dest: Path, timeout: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                f.write(chunk)


def _is_valid_jpg(path: Path) -> bool:
    """Comprueba que el fichero existe, no está vacío y empieza con magic bytes JPEG."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        header = f.read(3)
    return header == b"\xff\xd8\xff"


def download_image(
    image_id: int,
    url: str,
    output_dir: Path,
    retries: int = 3,
    timeout: int = 30,
) -> Tuple[int, bool, Optional[str]]:
    """
    Descarga una imagen y la guarda como <image_id>.jpg.

    Retorna: (image_id, success, error_message_or_None)
    """
    dest = output_dir / f"{image_id}.jpg"

    # Resume: si ya existe y es válida, omitir
    if _is_valid_jpg(dest):
        return image_id, True, None

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if _HAS_REQUESTS:
                _download_with_requests(url, dest, timeout)
            else:
                _download_with_urllib(url, dest, timeout)

            # Verificar que el fichero descargado es un JPEG válido
            if not _is_valid_jpg(dest):
                dest.unlink(missing_ok=True)
                raise ValueError("El fichero descargado no es un JPEG válido")

            return image_id, True, None

        except Exception as e:
            last_error = str(e)
            if dest.exists():
                dest.unlink(missing_ok=True)
            if attempt < retries:
                wait = 2 ** attempt  # backoff: 2s, 4s, 8s
                time.sleep(wait)

    return image_id, False, last_error


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_subset(path: Path) -> List[int]:
    """Lee el JSON del sampleo y devuelve la lista de image_ids."""
    with open(path) as f:
        data = json.load(f)
    ids = [entry["image_id"] for entry in data["images"]]
    log.info(f"Subset cargado: {len(ids)} imágenes de {path.name}")
    return ids


def load_url_map(image_data_path: Path, target_ids: set) -> Dict[int, str]:
    """
    Lee image_data.json y construye {image_id: url} solo para los ids necesarios.
    Hace streaming del JSON para no cargar todo en memoria si es grande.
    """
    log.info(f"Cargando URLs de {image_data_path.name} ...")
    with open(image_data_path) as f:
        data = json.load(f)

    url_map: Dict[int, str] = {}
    for entry in data:
        iid = entry["image_id"]
        if iid in target_ids:
            url = entry.get("url") or entry.get("image_url")
            if url:
                url_map[iid] = url

    found    = len(url_map)
    missing  = len(target_ids) - found
    log.info(f"  → URLs encontradas: {found} / {len(target_ids)}")
    if missing > 0:
        log.warning(f"  → Sin URL: {missing} imágenes (se omitirán)")

    return url_map


# ---------------------------------------------------------------------------
# Descarga en paralelo con progreso
# ---------------------------------------------------------------------------

class _FallbackProgress:
    """Barra de progreso mínima sin tqdm."""
    def __init__(self, total: int):
        self.total   = total
        self.n       = 0
        self._last   = -1

    def update(self, n: int = 1):
        self.n += n
        pct = int(100 * self.n / self.total)
        if pct != self._last and pct % 5 == 0:
            log.info(f"  Progreso: {self.n}/{self.total} ({pct}%)")
            self._last = pct

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def download_all(
    url_map:    Dict[int, str],
    output_dir: Path,
    workers:    int,
    retries:    int,
    timeout:    int,
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """
    Descarga todas las imágenes en paralelo.

    Retorna: (lista_ids_ok, lista_(id, error)_fallidos)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    succeeded:  List[int]                  = []
    failed:     List[Tuple[int, str]]      = []

    items = list(url_map.items())  # [(image_id, url), ...]
    ProgressCls = tqdm if _HAS_TQDM else _FallbackProgress
    progress_kwargs = (
        {"total": len(items), "unit": "img", "desc": "Descargando"} if _HAS_TQDM
        else {"total": len(items)}
    )

    with ProgressCls(**progress_kwargs) as pbar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_image, iid, url, output_dir, retries, timeout): iid
                for iid, url in items
            }

            for future in as_completed(futures):
                iid, ok, err = future.result()
                if ok:
                    succeeded.append(iid)
                else:
                    failed.append((iid, err))
                pbar.update(1)

    return succeeded, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Descarga imágenes de un subset estratificado de Visual Genome."
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="JSON de salida de vg_stratified_subset.py"
    )
    parser.add_argument(
        "--image_data",
        required=True,
        help="Ruta a image_data.json de VG (contiene las URLs)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Carpeta donde guardar las imágenes descargadas"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Descargas en paralelo (default: 8). Sube a 16-32 si tu red lo permite."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Reintentos por imagen en caso de fallo (default: 3)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout en segundos por imagen (default: 30)"
    )
    parser.add_argument(
        "--failed_log",
        default="vg_download_failed.json",
        help="Fichero donde guardar los ids que fallaron (default: vg_download_failed.json)"
    )
    args = parser.parse_args()

    if not _HAS_REQUESTS:
        log.warning(
            "requests no está instalado; se usará urllib (menos robusto). "
            "Instálalo con: pip install requests"
        )
    if not _HAS_TQDM:
        log.warning(
            "tqdm no está instalado; el progreso se mostrará cada 5%. "
            "Instálalo con: pip install tqdm"
        )

    # 1. Cargar subset y construir mapa de URLs
    target_ids = load_subset(Path(args.subset))
    url_map    = load_url_map(Path(args.image_data), set(target_ids))

    if not url_map:
        log.error("No se encontraron URLs para ningún image_id del subset. Abortando.")
        return

    # 2. Comprobar cuántas ya están descargadas (resume)
    output_dir = Path(args.output_dir)
    already_done = sum(
        1 for iid in url_map
        if _is_valid_jpg(output_dir / f"{iid}.jpg")
    )
    if already_done:
        log.info(f"Resume: {already_done} imágenes ya descargadas, se omitirán.")

    # 3. Descargar
    log.info(
        f"\nIniciando descarga: {len(url_map)} imágenes → {output_dir} "
        f"({args.workers} workers, {args.retries} reintentos)"
    )
    t0 = time.time()
    succeeded, failed = download_all(
        url_map, output_dir, args.workers, args.retries, args.timeout
    )
    elapsed = time.time() - t0

    # 4. Reporte
    log.info(f"\n── Resultado ──────────────────────────────")
    log.info(f"  ✓ Descargadas correctamente : {len(succeeded)}")
    log.info(f"  ✗ Fallidas                  : {len(failed)}")
    log.info(f"  Tiempo total                : {elapsed:.1f}s")
    if len(url_map) > 0:
        rate = len(url_map) / elapsed
        log.info(f"  Velocidad media             : {rate:.1f} imgs/s")

    # 5. Guardar log de fallos
    if failed:
        failed_path = Path(args.failed_log)
        with open(failed_path, "w") as f:
            json.dump(
                [{"image_id": iid, "error": err} for iid, err in failed],
                f,
                indent=2,
            )
        log.warning(
            f"\n  {len(failed)} imágenes fallidas guardadas en: {failed_path}\n"
            f"  Puedes reintentar con: python vg_download_subset.py "
            f"--subset {args.subset} --image_data {args.image_data} "
            f"--output_dir {args.output_dir}  (el resume las omitirá automáticamente)"
        )

    log.info(f"\n✓ Listo. Imágenes en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()