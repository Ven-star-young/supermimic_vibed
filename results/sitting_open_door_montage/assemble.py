"""Join the verified policy demos without changing their timing or frames."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parent
SEGMENTS = [
    (ROOT / 'sitting/sitting_grid.mp4', 'Sitting: 0, 8, 10, 13, 14, 19, 28, 34, 45'),
    (ROOT / 'doors/back_clean.mp4', 'Back door opening'),
    (ROOT / 'doors/side_clean.mp4', 'Side door opening'),
]


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)
    ]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true', help='Compatibility flag; clean clips are always used')
    parser.parse_args()
    segments = SEGMENTS
    suffix = '_clean'
    inputs, metadata, chapters = [], [';FFMETADATA1', 'title=G1 Sitting and Door Opening'], []
    elapsed = 0.0
    for path, title in segments:
        info = probe(path)
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height'], video['pix_fmt']) == (1920, 1080, 'yuv420p')
        assert video['avg_frame_rate'] == '25/1', (path, video['avg_frame_rate'])
        duration = float(info['format']['duration'])
        inputs.append("file '" + str(path).replace("'", "'\\''") + "'")
        metadata += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={round(elapsed * 1000)}',
                     f'END={round((elapsed + duration) * 1000)}', f'title={title}']
        chapters.append(dict(title=title, source=str(path), start_s=elapsed, duration_s=duration))
        elapsed += duration
    concat = ROOT / f'concat{suffix}.txt'
    concat.write_text('\n'.join(inputs) + '\n')
    ffmetadata = ROOT / f'chapters{suffix}.ffmetadata'
    ffmetadata.write_text('\n'.join(metadata) + '\n')
    output = ROOT / 'sitting_back_side_clean.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'warning', '-f', 'concat', '-safe', '0',
                    '-i', str(concat), '-f', 'ffmetadata', '-i', str(ffmetadata),
                    '-map', '0:v:0', '-map_metadata', '1', '-map_chapters', '1',
                    '-c:v', 'copy', '-an', '-movflags', '+faststart', str(output)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    final_info = probe(output)
    (ROOT / f'video_manifest{suffix}.json').write_text(json.dumps(dict(
        output=str(output), chapters=chapters, expected_duration_s=elapsed,
        probe=final_info, validation='Complete decode succeeded; segments retain original timing and frames.'
    ), indent=2) + '\n')
    print(output)
    print(f'{elapsed:.2f} s, 1920x1080, 25 fps')


if __name__ == '__main__':
    main()
