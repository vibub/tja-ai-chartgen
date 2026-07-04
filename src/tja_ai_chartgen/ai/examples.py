REFERENCE_EXAMPLES_PROMPT = r'''Reference chart examples, precomputed from study charts.
These examples are evenly sampled across each full chart, not only intros.
Use them only as style and audio-alignment examples. Do not copy note-string length;
the target output must always match each target bar's grids_per_bar.

[
  {
    "title": "\u30a2\u30b9\u30ce\u30e8\u30be\u30e9\u54e8\u6212\u73ed",
    "artist": "Orangestar feat.IA",
    "course": "Oni",
    "level": 9,
    "sample_strategy": "24 evenly sampled bars across the full chart",
    "total_chart_bars": 90,
    "total_audio_bars": 62,
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
        "bar": 9,
        "energy": 0.118,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          3,
          4,
          5,
          7,
          8,
          9,
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
          13
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000000000100000200000100000200000100000200200",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 12,
        "energy": 0.137,
        "grids_per_bar": 16,
        "onset_16": [
          4,
          5,
          7,
          8,
          12,
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
          12,
          15
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000200000100000200000100000200200100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 14,
        "energy": 0.163,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          5,
          7,
          9,
          11,
          12,
          13,
          15
        ],
        "accent_16": [
          0,
          12
        ],
        "beat_grids": [
          2,
          7,
          11,
          15
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000200000100000200000100000200200100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 17,
        "energy": 0.096,
        "grids_per_bar": 16,
        "onset_16": [
          3,
          4,
          5,
          8,
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
          3,
          7,
          11,
          14
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 20,
        "energy": 0.065,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4,
          5,
          8,
          9,
          10,
          13
        ],
        "accent_16": [
          4,
          8
        ],
        "beat_grids": [
          2,
          5,
          9,
          13
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "200000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 22,
        "energy": 0.125,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          4,
          5,
          8,
          9,
          10,
          13,
          14,
          15
        ],
        "accent_16": [
          4,
          8
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000200200100000200000100000200200100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 25,
        "energy": 0.095,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          4,
          5,
          8,
          9,
          10,
          13,
          15
        ],
        "accent_16": [
          4,
          8
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "500000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 28,
        "energy": 0.094,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          3,
          5,
          7,
          9,
          10,
          13,
          15
        ],
        "accent_16": [
          7
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000100100200000100100100000200000200000100100",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 30,
        "energy": 0.093,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          5,
          7,
          9,
          10,
          11,
          13,
          14,
          15
        ],
        "accent_16": [
          9
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000008000000700000000000000000800000300000300000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 33,
        "energy": 0.105,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          5,
          6,
          7,
          9,
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
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000100100200000100100100000200200200000100100",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 35,
        "energy": 0.124,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          3,
          5,
          6,
          7,
          8,
          10,
          12,
          14,
          15
        ],
        "accent_16": [
          8,
          12
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000000100000200000100100100000100100200000100100",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 38,
        "energy": 0.093,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          7,
          8,
          10,
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
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000008000000700000000000800000300000000000300000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 41,
        "energy": 0.097,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          7,
          8,
          10,
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
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000100100200000100100100000200200200000100100",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 43,
        "energy": 0.106,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          6,
          7,
          8,
          10,
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
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000000000000000000000000000000000000000000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 46,
        "energy": 0.091,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          3,
          4,
          7,
          8,
          9,
          12,
          13,
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
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000000100100200000100100100000100100200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 49,
        "energy": 0.109,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          3,
          4,
          7,
          8,
          9,
          11,
          12,
          13,
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
        "section": "break",
        "reference_notes": "200000100100200000100200100200100000200000300000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 51,
        "energy": 0.08,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          3,
          4,
          7,
          8,
          9,
          12,
          13,
          15
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "200000100100200000100100100000200200200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 54,
        "energy": 0.099,
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
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000000100100200000200200100000100100200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 57,
        "energy": 0.124,
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
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "200000100100200000100200100200100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 59,
        "energy": 0.089,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          4,
          6,
          8,
          9,
          12,
          13,
          14
        ],
        "accent_16": [
          4,
          8,
          12
        ],
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "100000100100200000100100200000100200200200100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 62,
        "energy": 0.0,
        "grids_per_bar": 16,
        "onset_16": [],
        "accent_16": [],
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "song_end",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "000000100100100000200000100100100000200000100100",
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
    "sample_strategy": "24 evenly sampled bars across the full chart",
    "total_chart_bars": 97,
    "total_audio_bars": 65,
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
        "bar": 9,
        "energy": 0.127,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          7,
          8,
          10,
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
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000100000200000200100000000100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 12,
        "energy": 0.058,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          4,
          7,
          12,
          15
        ],
        "accent_16": [
          4,
          12
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000000000200000100000000000100000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 15,
        "energy": 0.036,
        "grids_per_bar": 16,
        "onset_16": [
          6,
          9,
          12,
          14
        ],
        "accent_16": [
          12
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "500000000008000000000000100000200000200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 18,
        "energy": 0.008,
        "grids_per_bar": 16,
        "onset_16": [
          13
        ],
        "accent_16": [
          13
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "300000000000200000300000000000200000300000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 20,
        "energy": 0.019,
        "grids_per_bar": 16,
        "onset_16": [
          4,
          7
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
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "300000000000200000300000000000200000300000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 23,
        "energy": 0.007,
        "grids_per_bar": 16,
        "onset_16": [
          2
        ],
        "accent_16": [
          2
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
        "section": "break",
        "reference_notes": "500000000008000000100000200000200000100000300000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 26,
        "energy": 0.0,
        "grids_per_bar": 16,
        "onset_16": [],
        "accent_16": [],
        "beat_grids": [
          4,
          8,
          12,
          15
        ],
        "downbeat_grid": null,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000100100200000100100100000100000200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 29,
        "energy": 0.033,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          7,
          10,
          15
        ],
        "accent_16": [
          2
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
        "section": "break",
        "reference_notes": "100100100000200200200000100000400000400000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 32,
        "energy": 0.105,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4,
          7,
          9,
          10,
          12
        ],
        "accent_16": [
          4,
          12
        ],
        "beat_grids": [
          0,
          4,
          10,
          14
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000000000200000100000100100100000200200200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 34,
        "energy": 0.155,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
          7,
          8,
          9,
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
          2,
          5,
          9,
          12,
          15
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000100100200000100100100000100000200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 37,
        "energy": 0.165,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          3,
          4,
          5,
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
          3,
          6,
          10,
          14
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100100100000200200200000100000400000400000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 40,
        "energy": 0.171,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          3,
          5,
          6,
          7,
          9,
          10,
          11,
          13,
          14,
          15
        ],
        "accent_16": [
          9
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000000000200000100000100100100000200200200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 43,
        "energy": 0.084,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          4,
          6,
          8,
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
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000000100100100000200000100000100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 46,
        "energy": 0.015,
        "grids_per_bar": 16,
        "onset_16": [
          3,
          5
        ],
        "accent_16": [
          3
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
        "section": "break",
        "reference_notes": "300000000000200000300000000000200000300000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 48,
        "energy": 0.079,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          4,
          10,
          13,
          15
        ],
        "accent_16": [
          4
        ],
        "beat_grids": [
          2,
          6,
          10,
          13
        ],
        "downbeat_grid": 2,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "300000000000400000000000300000200200100000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 51,
        "energy": 0.02,
        "grids_per_bar": 16,
        "onset_16": [
          7,
          11,
          13
        ],
        "accent_16": [
          11
        ],
        "beat_grids": [
          1,
          5,
          9,
          13
        ],
        "downbeat_grid": 1,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "000000000000000008000000100000000000100000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 54,
        "energy": 0.007,
        "grids_per_bar": 16,
        "onset_16": [
          0
        ],
        "accent_16": [
          0
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
        "section": "break",
        "reference_notes": "500000000000000000000000000000000000000008000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 57,
        "energy": 0.016,
        "grids_per_bar": 16,
        "onset_16": [
          5,
          8
        ],
        "accent_16": [
          8
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
        "section": "break",
        "reference_notes": "100000000100200000100000500000000000000008000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 59,
        "energy": 0.006,
        "grids_per_bar": 16,
        "onset_16": [
          8
        ],
        "accent_16": [
          8
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
        "section": "outro",
        "reference_notes": "100100100000100100100000200200200000200200200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 62,
        "energy": 0.0,
        "grids_per_bar": 16,
        "onset_16": [],
        "accent_16": [],
        "beat_grids": [
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "400000400000400000400000000000200000300000300000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 65,
        "energy": 0.0,
        "grids_per_bar": 16,
        "onset_16": [],
        "accent_16": [],
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "song_end",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "100000100100100000100000100000100000100000100000",
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
    "sample_strategy": "24 evenly sampled bars across the full chart",
    "total_chart_bars": 72,
    "total_audio_bars": 38,
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
        "bar": 9,
        "energy": 0.275,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          3,
          5,
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
        "section": "break",
        "reference_notes": "100000200000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 11,
        "energy": 0.241,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          3,
          5,
          7,
          8,
          9,
          10,
          11,
          13,
          14,
          15
        ],
        "accent_16": [
          8
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
        "section": "break",
        "reference_notes": "100000000000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 12,
        "energy": 0.371,
        "grids_per_bar": 16,
        "onset_16": [
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
        "section": "break",
        "reference_notes": "100000000000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 14,
        "energy": 0.281,
        "grids_per_bar": 16,
        "onset_16": [
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
        "section": "break",
        "reference_notes": "100000000000100000200000100000000000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 15,
        "energy": 0.253,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          3,
          4,
          5,
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
        "section": "break",
        "reference_notes": "100000200000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 17,
        "energy": 0.288,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
          7,
          8,
          9,
          10,
          11,
          13,
          15
        ],
        "accent_16": [
          0,
          4,
          8
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
        "section": "break",
        "reference_notes": "100000200000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 19,
        "energy": 0.174,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
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
          0,
          4,
          8
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
        "section": "break",
        "reference_notes": "100000200000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 20,
        "energy": 0.223,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
          8,
          9,
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
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000000000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 22,
        "energy": 0.254,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          2,
          3,
          4,
          5,
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
          3,
          7,
          11,
          15
        ],
        "downbeat_grid": 3,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "break",
        "reference_notes": "100000000000100000200000100000100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 24,
        "energy": 0.191,
        "grids_per_bar": 16,
        "onset_16": [
          1,
          2,
          3,
          4,
          5,
          6,
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
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "break",
        "reference_notes": "100000000000100000200000100000100000200000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 25,
        "energy": 0.081,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          3,
          4,
          5,
          6,
          7,
          15
        ],
        "accent_16": [
          4
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
        "section": "break",
        "reference_notes": "100000200000100000200000100000000000200000200200",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 27,
        "energy": 0.029,
        "grids_per_bar": 16,
        "onset_16": [
          4,
          8,
          15
        ],
        "accent_16": [
          4,
          8
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
        "section": "break",
        "reference_notes": "100000200000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 28,
        "energy": 0.05,
        "grids_per_bar": 16,
        "onset_16": [
          2,
          4,
          6,
          8,
          15
        ],
        "accent_16": [
          4,
          8
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
        "section": "break",
        "reference_notes": "100000000000100000200000100000000000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 30,
        "energy": 0.06,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          4,
          5,
          9,
          13,
          15
        ],
        "accent_16": [
          0,
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
        "section": "break",
        "reference_notes": "100000100100100000100000100000100000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 32,
        "energy": 0.133,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          1,
          3,
          4,
          5,
          6,
          7,
          8,
          9,
          10,
          12,
          13,
          14
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
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "outro",
        "reference_notes": "100000100100100000100000100000100000200000000000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 33,
        "energy": 0.211,
        "grids_per_bar": 16,
        "onset_16": [
          0,
          2,
          4,
          6,
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
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_start",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "100000100000200000100000200000100000200000200200",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 35,
        "energy": 0.389,
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
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_middle",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "100000100000200000100000200000100000200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 36,
        "energy": 0.406,
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
          0,
          4,
          8,
          12
        ],
        "downbeat_grid": 0,
        "phrase_position": "phrase_end",
        "fill_candidate": true,
        "section": "outro",
        "reference_notes": "500000000008000000100000200000100100200000100000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      },
      {
        "bar": 38,
        "energy": 0.0,
        "grids_per_bar": 16,
        "onset_16": [],
        "accent_16": [],
        "beat_grids": [],
        "downbeat_grid": null,
        "phrase_position": "song_end",
        "fill_candidate": false,
        "section": "outro",
        "reference_notes": "100000200000100000200000100000100000100000200000",
        "reference_note_resolution": 48,
        "balloon_counts": []
      }
    ]
  }
]
'''


def get_reference_examples_prompt() -> str:
    return REFERENCE_EXAMPLES_PROMPT
