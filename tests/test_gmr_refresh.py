"""Replacing a motion prior under the same name must not reuse stale output."""

import hashlib

import numpy as np
import pytest

from motion_data import load_motion, save_motion
from motion_data.converters.gmr import convert_pair
from tests.test_open_door_motion import DOOR_XML, linear_door_clip


def test_same_name_updated_reference_requires_explicit_refresh(tmp_path):
    source = tmp_path / 'source' / 'prior.npz'
    output = tmp_path / 'prepared'
    clip = linear_door_clip()
    save_motion(source, clip)
    source.with_suffix('.xml').write_text(DOOR_XML)
    assert convert_pair(source, output)
    old_output = (output / source.name).read_bytes()
    clip['root_pos'][:, :2] += [.1, -.2]
    save_motion(source, clip)
    source_bytes = source.read_bytes()
    with pytest.raises(ValueError, match='Source pair changed'):
        convert_pair(source, output)
    assert (output / source.name).read_bytes() == old_output
    assert convert_pair(source, output, overwrite=True)
    prepared = load_motion(output / source.name)
    np.testing.assert_array_equal(prepared['root_pos'], clip['root_pos'])
    np.testing.assert_array_equal(prepared['door_angle'], clip['door_angle'])
    assert str(prepared['source_sha256']) == hashlib.sha256(source_bytes).hexdigest()
    assert source.read_bytes() == source_bytes
    assert not convert_pair(source, output)


def test_same_name_updated_xml_and_legacy_provenance_are_not_silently_skipped(tmp_path):
    source = tmp_path / 'source' / 'prior.npz'
    output = tmp_path / 'prepared'
    save_motion(source, linear_door_clip())
    source.with_suffix('.xml').write_text(DOOR_XML)
    convert_pair(source, output)
    source.with_suffix('.xml').write_text(DOOR_XML.replace('mass="8"', 'mass="9"'))
    with pytest.raises(ValueError, match='Source pair changed'):
        convert_pair(source, output)
    convert_pair(source, output, overwrite=True)
    assert (output / 'prior.xml').read_bytes() == source.with_suffix('.xml').read_bytes()
    prepared = load_motion(output / source.name)
    prepared.pop('source_sha256')
    save_motion(output / source.name, prepared)
    with pytest.raises(ValueError, match='no source hashes'):
        convert_pair(source, output)
