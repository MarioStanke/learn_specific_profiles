""" Functions to convert positions between fwd/rev DNA and protein sequences """

def fwd_to_rc(pos: int, seqlen: int) -> int:
    """ Convert position in forward sequence to position in reverse complement sequence """
    assert seqlen > 0, "[fwd_to_rc] Sequence length must be positive"
    assert 0 <= pos < seqlen, f"[fwd_to_rc] Position {pos} out of range [0, {seqlen})"
    return seqlen - pos - 1


def rc_to_fwd(pos: int, seqlen: int) -> int:
    """ Convert position in reverse complement sequence to position in forward sequence """
    return fwd_to_rc(pos, seqlen) # same operation


def dna_to_aa(pos: int) -> tuple[int, int]:
    """ Convert position in a DNA sequence to position in protein sequence. Returns a tuple (frame_idx, aa_pos).
     It's up to the caller to determine if `pos` comes from a fwd or rc sequence. For a six-frame-translation setting,
      add 3 to the returned frame manually if `pos` comes from a reverse complement sequence! """
    assert 0 <= pos, f"[dna_to_aa] Position {pos} must be positive"
    frame = pos % 3
    aa_pos = (pos - frame) / 3
    assert aa_pos == int(aa_pos), f"[dna_to_aa] Failed to convert position {pos} to integer, got {aa_pos} in {frame=}"
    return frame, int(aa_pos)


def aa_to_dna(frame_idx: int, aa_pos: int) -> int:
    """ Convert position in a protein sequence to position in DNA sequence. Only frames in [0,3) are allowed, i.e. this
    function is agnostic of six-frame-translation settings. If your frame is >=3, subtract 3 from it before calling this
    function and consider using rc_to_fwd() to convert the returned position back to the equivalent forward sequence 
    position. """
    assert 0 <= frame_idx < 3, f"[aa_to_dna] Invalid frame index {frame_idx}"
    assert 0 <= aa_pos, f"[aa_to_dna] Position {aa_pos} must be positive"
    return aa_pos * 3 + frame_idx


def dna_range_to_aa(dna_start: int, dna_end: int, frame_idx: int, partial_overlap:bool=True) -> tuple[int, int]:
    """ Convert a range in a DNA sequence to a range in a translated sequence. Returns a tuple (aa_start, aa_end). Range
    end follows Python convention, i.e. it's exclusive (dna_end is assumed to be one after the last base in the DNA 
    range, and aa_end is one after the last aa that covers the equivalent range). Only frames in [0,3) are allowed, i.e.
    this function is agnostic of six-frame-translation settings. If your frame is >=3, make sure the DNA range is
    converted to the reverse complement range, and subtract 3 from the frame before calling this function.

    If `partial_overlap` is set to `True`, any base triplet that has at least one base inside the DNA range is 
    considered to result in an amino acid that belongs to the equivalent aa range. If `partial_overlap` is `False`,
    only base triplets that are fully inside the DNA range are considered to result in an amino acid that belongs to
    the equivalent aa range. """
    assert dna_start <= dna_end, f"[dna_range_to_aa] Invalid range [{dna_start}, {dna_end}]"
    assert frame_idx in (0, 1, 2), f"[dna_range_to_aa] Invalid frame index {frame_idx}"

    start_frame = dna_start % 3
    end_frame = dna_end % 3
    
    start_delta = frame_idx - start_frame
    end_delta = frame_idx - end_frame
    assert start_delta in range(-2,3), f"[dna_range_to_aa] Invalid start frame {start_frame} for {frame_idx=}"
    assert end_delta in range(-2,3), f"[dna_range_to_aa] Invalid end frame {end_frame} for {frame_idx=}"

    if partial_overlap:
        if start_delta == 0:
            aa_start  = (dna_start - frame_idx) / 3
        elif start_delta  in [-2, 1]:
            aa_start = (dna_start - frame_idx - 2) / 3
        else: # start_delta in [-1, 2]
            aa_start = (dna_start - frame_idx - 1) / 3

        if end_delta == 0:
            aa_end = (dna_end - frame_idx) / 3
        elif end_delta  in [-2, 1]:
            aa_end = (dna_end - frame_idx - 2) / 3
        else: # end_delta in [-1, 2]
            aa_end = (dna_end - frame_idx - 1) / 3

    else: # full overlap required
        if start_delta == 0:
            aa_start  = (dna_start - frame_idx) / 3
        elif start_delta  in [-2, 1]:
            aa_start = (dna_start - frame_idx + 1) / 3
        else: # start_delta in [-1, 2]
            aa_start = (dna_start - frame_idx + 2) / 3

        if end_delta == 0:
            aa_end = (dna_end - frame_idx) / 3
        elif end_delta  in [-2, 1]:
            aa_end = (dna_end - frame_idx - 5) / 3
        else: # end_delta in [-1, 2]
            aa_end = (dna_end - frame_idx - 4) / 3

    assert aa_start == int(aa_start), \
        f"[dna_range_to_aa] Failed to convert start position {dna_start} to integer, got {aa_start}" \
        + f" with {start_delta=}, {frame_idx=}"
    assert aa_end == int(aa_end), \
        f"[dna_range_to_aa] Failed to convert end position {dna_end} to integer, got {aa_end}" \
        + f" with {end_delta=}, {frame_idx=}"
    
    return aa_start, aa_end