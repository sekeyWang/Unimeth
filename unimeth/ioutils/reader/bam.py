"""
BAM file reader with indexing support.

Provides efficient random access to BAM records and methylation prediction loading.
"""
import os
import hashlib
import pickle
from typing import Dict, List

import pysam

from unimeth.utils import local_print
from unimeth.utils.bam_tags import get_mod_config, get_target_positions
from unimeth.data.coords import get_ref_pos


BAM_INDEX_SUFFIX = ".unimeth.idx"


def default_bam_index_file(bam_path: str) -> str:
    """Return the preferred persistent UniMeth index path next to the BAM file."""
    return f"{bam_path}{BAM_INDEX_SUFFIX}"


def _bam_index_hash(bam_path: str) -> str:
    absolute_path = os.path.abspath(bam_path)
    try:
        stat = os.stat(bam_path)
        payload = f"{absolute_path}:{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        payload = absolute_path
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _can_write_index_directory(index_path: str) -> bool:
    index_dir = os.path.dirname(os.path.abspath(index_path)) or "."
    probe_path = os.path.join(index_dir, f".unimeth-index-write-test.{os.getpid()}")
    try:
        fd = os.open(probe_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        os.remove(probe_path)
        return not os.path.exists(index_path) or os.access(index_path, os.W_OK)
    except OSError:
        try:
            if os.path.exists(probe_path):
                os.remove(probe_path)
        except OSError:
            pass
        return False


def _fallback_bam_index_file(bam_path: str, cache_dir: str = None) -> str:
    root_dir = cache_dir or os.getcwd()
    filename = os.path.basename(os.path.normpath(bam_path))
    digest = _bam_index_hash(bam_path)
    return os.path.join(root_dir, "data_cache", "bam_index", f"{filename}.{digest}{BAM_INDEX_SUFFIX}")


def resolve_bam_index_file(
    bam_path: str,
    cache_dir: str = None,
    require_writable: bool = True,
) -> tuple[str, bool]:
    """
    Resolve the UniMeth BAM index file.

    Returns:
        Tuple of (index_path, is_temporary). Persistent indexes live next to the
        BAM file; temporary fallback indexes live under cache_dir/data_cache.
    """
    preferred_index = default_bam_index_file(bam_path)
    if not require_writable and os.path.exists(preferred_index):
        return preferred_index, False
    if _can_write_index_directory(preferred_index):
        return preferred_index, False
    return _fallback_bam_index_file(bam_path, cache_dir=cache_dir), True


def _index_is_stale(bam_path: str, index_path: str) -> bool:
    if not os.path.exists(index_path):
        return True
    try:
        return os.path.getmtime(index_path) < os.path.getmtime(bam_path)
    except OSError:
        return True


def cleanup_bam_index(index_path: str, is_temporary: bool, created_by_this_run: bool) -> None:
    """Remove a temporary fallback index that was created by this run."""
    if not (index_path and is_temporary and created_by_this_run):
        return
    try:
        os.remove(index_path)
        parent = os.path.dirname(index_path)
        try:
            data_cache_dir = os.path.dirname(parent)
            if os.path.basename(parent) == "bam_index" and os.path.basename(data_cache_dir) == "data_cache":
                os.rmdir(parent)
                os.rmdir(data_cache_dir)
        except OSError:
            pass
    except FileNotFoundError:
        pass
    except OSError as exc:
        local_print(f"Warning: failed to remove temporary BAM index {index_path}: {exc}")


class BamReader:
    """
    Indexed BAM file reader for random access by read ID.
    
    Automatically builds and caches an index mapping read IDs to file positions
    for efficient retrieval.
    
    Attributes:
        bam_file: Open pysam AlignmentFile handle
        bam_index: Dictionary mapping read_id -> list of file positions
    """
    
    def __init__(
        self,
        bam_path: str,
        force_rebuild_index: bool = False,
        index_cache_dir: str = None,
        allow_index_build: bool = True,
    ):
        """
        Initialize indexed BAM reader.
        
        Args:
            bam_path: Path to BAM file
            force_rebuild_index: If True, rebuild index even if cached index exists
            index_cache_dir: Directory for temporary fallback index cache
            allow_index_build: If False, missing or invalid indexes raise an error
        """
        self.bam_path = bam_path
        self.bam_file = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
        self.bam_index_created = False

        preferred_index = default_bam_index_file(bam_path)
        require_writable = (
            force_rebuild_index
            or not os.path.exists(preferred_index)
            or _index_is_stale(bam_path, preferred_index)
        )
        self.bam_index_file, self.bam_index_is_temporary = resolve_bam_index_file(
            bam_path,
            cache_dir=index_cache_dir,
            require_writable=require_writable,
        )

        if (
            force_rebuild_index
            or not os.path.exists(self.bam_index_file)
            or _index_is_stale(bam_path, self.bam_index_file)
        ):
            if not allow_index_build:
                raise FileNotFoundError(
                    f"BAM index is missing or stale: {self.bam_index_file}. "
                    "The main inference process must build it before workers start."
                )
            self.bam_index = self._build_bam_index(self.bam_index_file)
            self.bam_index_created = True
        else:
            try:
                with open(self.bam_index_file, 'rb') as fr:
                    self.bam_index = pickle.load(fr)
            except Exception as exc:
                if not allow_index_build:
                    raise RuntimeError(f"Failed to load BAM index {self.bam_index_file}") from exc
                self.bam_index_file, self.bam_index_is_temporary = resolve_bam_index_file(
                    bam_path,
                    cache_dir=index_cache_dir,
                    require_writable=True,
                )
                self.bam_index = self._build_bam_index(self.bam_index_file)
                self.bam_index_created = True
    

    def _build_bam_index(self, bam_index_file: str) -> dict:
        local_print('Building bam index...')
        self.bam_file.reset()
        bam_index = {}
        read_ptr = self.bam_file.tell()
        
        for bam_read in self.bam_file:
            if bam_read.is_supplementary or bam_read.is_secondary:
                read_ptr = self.bam_file.tell()
                continue
            
            if bam_read.has_tag('pi'):
                # Index by pi (parent read ID) for SignalReader lookup
                pi_id = bam_read.get_tag("pi")
                if pi_id not in bam_index:
                    bam_index[pi_id] = []
                bam_index[pi_id].append(read_ptr)
                # Also index by query_name so BAM writer can find split reads
                qname = bam_read.query_name
                if qname not in bam_index:
                    bam_index[qname] = []
                bam_index[qname].append(read_ptr)
            else:
                read_id = bam_read.query_name
                if read_id not in bam_index:
                    bam_index[read_id] = []
                bam_index[read_id].append(read_ptr)
            read_ptr = self.bam_file.tell()
        
        index_dir = os.path.dirname(os.path.abspath(bam_index_file))
        os.makedirs(index_dir, exist_ok=True)
        temp_index_file = f"{bam_index_file}.tmp.{os.getpid()}.{id(bam_index)}"
        try:
            with open(temp_index_file, 'wb') as fw:
                pickle.dump(bam_index, fw)
            os.replace(temp_index_file, bam_index_file)
        finally:
            try:
                if os.path.exists(temp_index_file):
                    os.remove(temp_index_file)
            except OSError:
                pass
        
        local_print(f'Bam index built and saved in {bam_index_file}. Size: {len(bam_index)}')
        return bam_index
    
    def get_read_by_id(self, read_id: str) -> list:
        """
        Get BAM records by read ID.
        
        Args:
            read_id: Read ID to look up
            
        Returns:
            List of pysam AlignedSegment objects
        """
        bam_reads = []
        if read_id not in self.bam_index:
            return bam_reads
        
        read_ptrs = self.bam_index[read_id]
        for read_ptr in read_ptrs:
            self.bam_file.seek(read_ptr)
            bam_read = next(self.bam_file)
            bam_reads.append(bam_read)
        
        return bam_reads
    
    def load_site_predictions(
        self,
        mod_type: str = 'm'
    ) -> Dict[str, List[float]]:
        """
        Load site-level methylation predictions from BAM MM/ML tags.

        Args:
            mod_type: Modification type ('m' for CpG/CHG/CHH, 'a' for m6A)

        Returns:
            Dictionary mapping chr_pos to list of predictions
        """
        from tqdm import tqdm
        from unimeth.utils.bam_tags import get_modifications

        site: Dict[str, List[float]] = {}
        base, mod_key = get_mod_config(mod_type)

        # Sequential scan is much faster than indexed random access
        self.bam_file.reset()
        seen_reads: set = set()
        for bam_read in tqdm(self.bam_file, desc='Reading BAM'):
            if bam_read.is_supplementary or bam_read.is_secondary:
                continue
            if bam_read.query_name in seen_reads:
                continue
            seen_reads.add(bam_read.query_name)
            if bam_read.get_forward_sequence() is None:
                continue
            if not bam_read.has_tag('ML') or len(bam_read.get_tag('ML')) == 0:
                continue

            seq = bam_read.get_forward_sequence().upper()
            pred_pos = get_target_positions(seq, base)

            aligned_pairs = bam_read.get_aligned_pairs()
            if len(aligned_pairs) == 0:
                continue

            if bam_read.is_reverse:
                aligned_pairs = aligned_pairs[::-1]

            ref_pos_list = get_ref_pos(seq, pred_pos, aligned_pairs, bam_read.is_reverse)
            pos_to_ref = {pred_pos[i]: ref_pos_list[i] for i in range(len(pred_pos))}

            chrom = bam_read.reference_name
            mod = get_modifications(bam_read, mod_key)

            for pos, label in mod.items():
                pred_value = label / 255
                ref_pos = pos_to_ref.get(pos, -1)
                if ref_pos == -1:
                    continue

                name = f'{chrom}_{ref_pos}'
                if name not in site:
                    site[name] = []
                site[name].append(pred_value)

        return site
