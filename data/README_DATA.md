# Data

The notebook reads two processed, ready-to-use files from this folder and never downloads anything. How they were built from the raw agency records is covered in a companion project, *Building Champi's Track File* (planned for December 2026).

## champi_2026_track.csv (required)

Champi's track, one row per 6-hourly position, from official genesis (14 Oct 2026 12:00 UTC) to its exit north of Taiwan (24 Oct 12:00 UTC). 41 rows, no gaps in time.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `iso_time` | date-time | UTC | Time of the position |
| `name` | text | | Storm name (CHAMPI) |
| `season` | integer | year | Typhoon season (2026) |
| `lat` | decimal | degrees north | Latitude of the storm's center |
| `lon` | decimal | degrees east | Longitude of the storm's center |
| `wind_kt` | integer | knots | Maximum 1-minute sustained wind near the center (JTWC estimate) |
| `pres_hpa` | integer | hPa | Minimum central pressure |
| `poci_hpa` | integer | hPa | Pressure of the outermost closed isobar (the storm's edge pressure) |
| `roci_nm` | integer | nautical miles | Radius of the outermost closed isobar: roughly the edge of the whole storm |
| `rmw_nm` | integer | nautical miles | Radius of maximum winds: roughly the eyewall |
| `r34_ne_nm`, `r34_se_nm`, `r34_sw_nm`, `r34_nw_nm` | decimal | nautical miles | How far gale-force winds (34 kt or more) reach in each quadrant. Blank when they did not reach that quadrant: all four while Champi was below 34 kt, and one or two quadrants in its first days as a tropical storm |
| `source` | text | | Which version of the track this is (see below) |

1 nautical mile = 1.852 km; 1 knot = 1.852 km/h. The notebook converts all radii to km.

**Completeness.** Every column is filled in every row except the four gale-force radii (7, 4, 4 and 6 blanks in NE, SE, SW and NW).

**Versions**
- `jtwc_bdeck_working_best_track` (current): the Joint Typhoon Warning Center's working best track for 29W, posted as an ATCF b-deck by UCAR RAL. Provisional: the agencies finalize the best track months after the season.
- Earlier versions (RainViewer feed, with three JTWC advisory positions) are kept in `archive/` for the record.

**Cross-check:** the IBTrACS ACTIVE file (downloaded 25 Oct 2026) covers 14 Oct 12:00 to 22 Oct 00:00 UTC. Its positions and pressures match the b-deck exactly; its winds are within 2 kt.

**Sources**
- JTWC working best track: https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/2026/bwp292026.dat
- IBTrACS v04r01: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ (file `ibtracs.ACTIVE.list.v04r01.csv`). Dataset DOI: 10.25921/82ty-9e16. Reference: Knapp et al. (2010), Bull. Amer. Meteor. Soc. 91, 363-376, doi 10.1175/2009BAMS2755.1.

## land_wpac.geojson (required for maps)

Land outlines (polygons in longitude and latitude) for every coast that touches the western Pacific map, used to draw the maps and to measure distance to Taiwan's coast.

**Original source:** Natural Earth 1:50m land, public domain
https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson

