REFERENCE_EXAMPLES_PROMPT = r'''Reference chart examples, precomputed from bundled study charts.
Use these only as style and audio-alignment examples. Do not copy note-string length;
the target output must always match each target bar's grids_per_bar.

[
  {
    "title": "\u30a2\u30b9\u30ce\u30e8\u30be\u30e9\u54e8\u6212\u73ed",
    "artist": "Orangestar feat.IA",
    "course": "Oni",
    "level": 9,
    "bars": [
      {
        "bar": 1,
        "energy": 0.22,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 2,
        "energy": 0.321,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          12,
          14,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          0,
          4,
          8,
          12,
          15
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 3,
        "energy": 0.212,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          12,
          14,
          15
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 4,
        "energy": 0.237,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          12,
          14,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 5,
        "energy": 0.254,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          12,
          13,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 6,
        "energy": 0.156,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          5,
          6,
          7,
          10,
          11,
          14,
          15
        ],
        "accent_16": [
          0,
          4
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 7,
        "energy": 0.111,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          13
        ],
        "accent_16": [
          0,
          4,
          8
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 8,
        "energy": 0.168,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          6,
          7,
          8,
          10,
          11,
          12,
          14
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "song_end",
        "fill_candidate": true,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      }
    ]
  },
  {
    "title": "\u30b0\u30c3\u30d0\u30a4\u5ba3\u8a00",
    "artist": "Chinozo feat. flower",
    "course": "Oni",
    "level": 8,
    "bars": [
      {
        "bar": 1,
        "energy": 0.132,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          5,
          7,
          8,
          9,
          11,
          15
        ],
        "accent_16": [
          0,
          8
        ],
        "beat_grids": [
          0,
          5,
          9,
          13
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "400000000000400000000000100000200000200200200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 2,
        "energy": 0.083,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          3,
          5,
          7,
          8,
          9,
          11,
          13,
          14,
          15
        ],
        "accent_16": [
          8
        ],
        "beat_grids": [
          1,
          5,
          9,
          13,
          15
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000100000200000100000000000100100200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 3,
        "energy": 0.086,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4,
          5,
          6,
          8,
          9,
          10,
          12,
          14
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [
          4,
          8,
          12,
          15
        ],
        "downbeat_grid": null,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000100000000000100000100200100200100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 4,
        "energy": 0.05,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          8,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          8,
          12
        ],
        "beat_grids": [
          4,
          8,
          12,
          15
        ],
        "downbeat_grid": null,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "intro",
        "reference_notes": "100000200200200000100000000000100000100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 5,
        "energy": 0.089,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          4,
          6,
          8,
          9,
          10,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [
          4,
          8,
          12,
          15
        ],
        "downbeat_grid": null,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000100000200000100000200000100000200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 6,
        "energy": 0.085,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          5,
          6,
          11,
          13,
          14,
          15
        ],
        "accent_16": [
          13
        ],
        "beat_grids": [
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000100000200000100000000000100100200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 7,
        "energy": 0.107,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          5,
          6,
          7,
          10,
          11,
          13,
          14,
          15
        ],
        "accent_16": [
          10
        ],
        "beat_grids": [
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000100000000000100000100200100200100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 8,
        "energy": 0.122,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          7,
          8,
          9,
          10,
          11,
          12,
          15
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [
          4,
          8,
          12
        ],
        "downbeat_grid": null,
        "phrase_position": "song_end",
        "fill_candidate": true,
        "section": "intro",
        "reference_notes": "100000200200200000100000000000100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      }
    ]
  },
  {
    "title": "\u547d\u306b\u5acc\u308f\u308c\u3066\u3044\u308b\u3002",
    "artist": "\u30ab\u30f3\u30b6\u30ad\u30a4\u30aa\u30ea feat.\u521d\u97f3\u30df\u30af",
    "course": "Edit",
    "level": 8,
    "bars": [
      {
        "bar": 1,
        "energy": 0.061,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4,
          7,
          10,
          12,
          15
        ],
        "accent_16": [
          4,
          12
        ],
        "beat_grids": [
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000200000000100000100000000200000000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 2,
        "energy": 0.046,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4,
          14
        ],
        "accent_16": [
          4
        ],
        "beat_grids": [
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 3,
        "energy": 0.076,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          4,
          10,
          12,
          15
        ],
        "accent_16": [
          0,
          4,
          12
        ],
        "beat_grids": [
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000000000100000200000000100000000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 4,
        "energy": 0.035,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4
        ],
        "accent_16": [
          4
        ],
        "beat_grids": [
          0,
          4,
          8,
          11,
          15
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "intro",
        "reference_notes": "100000000000000000000000000000000000",
        "reference_note_resolution": 36,
        "balloon_counts": []
      },
      {
        "bar": 5,
        "energy": 0.357,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          4,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000100000200000000000",
        "reference_note_resolution": 24,
        "balloon_counts": []
      },
      {
        "bar": 6,
        "energy": 0.361,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000000000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 7,
        "energy": 0.423,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "intro",
        "reference_notes": "100000200000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 8,
        "energy": 0.411,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          11,
          12,
          13,
          14,
          15
        ],
        "accent_16": [
          0,
          4,
          8,
          12
        ],
        "beat_grids": [
          2,
          6,
          10,
          14
        ],
        "downbeat_grid": 2,
        "phrase_position": "song_end",
        "fill_candidate": true,
        "section": "intro",
        "reference_notes": "100000000000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      }
    ]
  }
]
'''


def get_reference_examples_prompt() -> str:
    return REFERENCE_EXAMPLES_PROMPT
