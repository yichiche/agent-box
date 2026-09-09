# Plot Tool CSV Format

Plot Tool has a small CSV contract for generic Pareto data. It is separate from
the benchmark-oriented import formats in the InferenceX Curve workspace.

Plot Tool accepts local `.csv` files only. It does not accept TSV, JSON, JSONL,
ZIP, raw InferenceX benchmark exports, or GitHub Actions artifacts.

## Canonical header

Use this header when generating a Plot Tool file:

```csv
Line ID,Line Name,X,Y,Color,Line Type,Line Marker,Layer
```

`Line ID`, `Line Name`, `X`, and `Y` are required. The remaining columns are
optional. Input columns may appear in a different order, and header matching is
case-insensitive with surrounding and repeated whitespace ignored. Exported
files always use the canonical order shown above.

| Column | Required | Description | Default when omitted or empty |
| --- | --- | --- | --- |
| `Line ID` | Yes | Stable, non-empty identifier. Rows with the same ID become points in one curve. | None |
| `Line Name` | Yes | Non-empty legend and tooltip name for the curve. | None |
| `X` | Yes | Finite numeric X coordinate. | None |
| `Y` | Yes | Finite numeric Y coordinate. | None |
| `Color` | No | Valid CSS color, preferably a six-digit hex value such as `#4e79a7`. | Next color in the Plot Tool default palette |
| `Line Type` | No | `solid`, `dashed`, `dotted`, `dashdot`, or `long-dash`. | `solid` |
| `Line Marker` | No | `circle`, `square`, `triangle`, `diamond`, `star`, `plus`, or `cross`. | `circle` |
| `Layer` | No | Finite numeric render layer. Curves with larger values render later and appear above lower layers. | Import order, starting at `1` |

`X`, `Y`, and `Layer` accept ordinary decimal or scientific notation as long as
the parsed value is finite. Zero and negative X/Y values are valid CSV data, but
Plot Tool will refuse to enable Log Scale for an axis while any visible value on
that axis is zero or negative.

## Curve grouping and metadata consistency

Each CSV row represents one point. Repeat the same `Line ID` to add multiple
points to a curve. For all rows with that ID, repeat the same line metadata:

- `Line Name`
- `Color`
- `Line Type`
- `Line Marker`
- `Layer`

This includes empty optional values. For example, do not provide a color on the
first row for a line and leave it empty on later rows. If metadata differs, the
entire file is rejected and the error identifies the inconsistent field, the
current row, and the first row for that `Line ID`.

Point rows remain in CSV order in the editor. Pareto calculation is performed
per curve, and the rendered Pareto path is connected in ascending numeric X
order regardless of row order.

## Complete example

```csv
Line ID,Line Name,X,Y,Color,Line Type,Line Marker,Layer
service-a,Service A,10,220,#4e79a7,solid,circle,1
service-a,Service A,20,310,#4e79a7,solid,circle,1
service-a,Service A,30,355,#4e79a7,solid,circle,1
service-b,Service B,12,260,#f28e2c,dashed,square,2
service-b,Service B,25,340,#f28e2c,dashed,square,2
service-b,Service B,38,370,#f28e2c,dashed,square,2
```

A minimal file may omit every style column:

```csv
Line ID,Line Name,X,Y
experiment-1,Experiment 1,1,10
experiment-1,Experiment 1,2,17.5
experiment-1,Experiment 1,4,23
```

## CSV quoting

The delimiter is a comma. UTF-8 files with or without a byte-order mark (BOM)
are supported, and empty rows are ignored. Use standard double-quoted fields
when a value contains a comma, quote, or line break; represent a literal quote
by doubling it:

```csv
Line ID,Line Name,X,Y
quoted-example,"Service ""A"", production",1,10
quoted-example,"Service ""A"", production",2,18
```

## Import and conflict handling

1. Open the `Plot Tool` workspace and select `Import CSV`.
2. Choose a `.csv` file. Parsing and validation happen before workspace data is
   changed.
3. Review the detected curve and point counts, then select the curves to add.
4. For every selected `Line ID` already present in the workspace, explicitly
   enable `Replace existing line`.
5. Select `Add Selected` to commit the import, or `Cancel` to discard the
   preview without changing the workspace.

Conflicting IDs are never renamed or merged automatically. A selected conflict
cannot be submitted until `Replace existing line` is enabled. Non-conflicting
curves are appended and made visible. Replacement changes the entire curve,
including all of its points and style metadata.

The importer rejects, among other cases:

- a missing required header;
- an empty `Line ID` or `Line Name`;
- an empty, infinite, `NaN`, or otherwise non-numeric X/Y value;
- a non-numeric `Layer` value;
- inconsistent metadata within a `Line ID`;
- a file with no point rows;
- an unterminated quoted field.

## Export and round trips

`Download CSV` exports the canonical eight columns in this exact order:

```text
Line ID, Line Name, X, Y, Color, Line Type, Line Marker, Layer
```

The exporter considers every curve in the Plot Tool workspace, including
currently hidden curves. Every point with finite X and Y values is included;
incomplete or invalid draft rows are skipped. A curve with no finite point has
no CSV row and therefore cannot be reconstructed from that export. The
downloaded filename is `pareto-plot-YYYY-MM-DD.csv`.

An exported file can be imported again without losing curve IDs, names, finite
points, colors, line types, markers, or layers. The CSV intentionally does not
contain chart-level or UI state, including:

- title, subtitle, axis titles, or watermark;
- X/Y optimization goals;
- axis Log Scale settings;
- curve visibility, search, or editor collapsed state;
- `Optimal Only`, `Line Labels`, or `Better Direction` switch values.

Those settings remain part of the current Plot Tool workspace rather than the
CSV data contract.
