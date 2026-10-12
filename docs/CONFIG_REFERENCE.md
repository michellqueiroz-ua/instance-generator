# JSON configuration reference

This reference describes the configuration paths implemented by `REQreate/input_json.py` and `REQreate/passenger_requests.py`, and calls out spellings that occur in example files but are not accepted by that parser. Configurations are JSON objects. The parser requires `network` and `attributes`; in practice, include `problem` and `requests` as well.

## Top-level keys

| Key | Meaning and default |
| --- | --- |
| `network` | **Required.** OSM place name used to load or download the street network. |
| `problem` | Problem label copied into the generated instance. No parser default; include it. The UI's Patient Transport preset uses `"DARP"`. |
| `seed` | Random seed for Python and NumPy. Defaults to `1`. |
| `replicas` | Number of generated replicas. Defaults to `1`. |
| `requests` | Number of requests to generate. No useful default; include it. |
| `attributes` | **Required.** Ordered attribute definitions; dependency order is computed from expressions and constraints. |
| `parameters` | List of named configuration values, location/zone collections, or generation flags. It can be empty, but include `"parameters": []` when attributes use `type: "location"` so the parser initializes its internal location registry. |
| `places` | Optional list of named custom locations and zones; see [Places and zones](#places-and-zones). |
| `locations` | Seen in a few basic examples, but **not read by `input_json.py`**. The parser's corresponding key is `places`; entries under `locations` are ignored by this parser. |
| `instance_filename` | List of parameter/property names used to construct the output filename. The parser only sets it when present, and later generation iterates it; include it in a usable configuration. Names are checked against available parameters and the properties `dynamism`, `urgency`, and `geographic_dispersion`. |
| `max_speed_factor` | Maximum speed multiplier used when deriving vehicle speed from the network. Defaults to `0.5` when creating a new network; ignored when a cached network is loaded. |
| `get_fixed_lines` | Optional fixed-line/network retrieval option passed to the network downloader; defaults to `null` (not requested). Example value: `"deconet"`. |
| `set_fixed_speed` | Optional object setting a constant vehicle speed. `vehicle_speed_data` is numeric; `vehicle_speed_data_unit` is `mps`, `kmh`, or `miph`. The speed is converted to metres/second. |
| `point` | Optional object with `lon` and `lat` for downloading a network around a point. `dist` is then read as the download radius/distance. `network` is still required. |
| `dist` | Distance passed to the network downloader when `point` is present. |
| `travel_time_matrix` | Optional list of place/attribute names for which to produce travel-time-matrix output. Each name must be registered as a location. |
| `method_pois` | Optional list of methods; the current parser uses the first item. Its `locations` is a two-element list of origin/destination attribute names, and its `pdf` object specifies the distance distribution used to place the destination relative to the origin. |

`problem` values in the examples include `DARP`, `ODBRP`, and `PRICE`. The parser copies the value rather than validating it. The code's lower-level `Instance.set_problem_type` recognizes `DARP`, `ODBRP`, `ODBRPFL`, and `SBRP`.

## Places and zones

The parser reads custom entries from `places`. Each entry has a `name` and `type`:

* `{"type": "location", ...}` defines a fixed point. Supply `lon` and `lat`, or set `"centroid": true` to use the network polygon's centroid. Coordinates must fall within the network boundary. Optional `class` is one of `school`, `coordinate`, or `bus_stop`; `school` and `bus_stop` add the location to the corresponding network data.
* `{"type": "zone", ...}` defines an area used to sample locations. Supply `lon` and `lat`, or `"centroid": true`, for its center. `length_lon` and `length_lat` define a rectangular zone's dimensions and must both be positive to form a rectangle. `radius` defines a circular zone instead; the code does not allow a positive radius together with a positive length. Dimensions default to `0`, and `length_unit` defaults to metres (`m`). Supported length units are `m`, `km`, and `mi`; the lengths are converted to metres. `radius` is passed through in metres.

Only `places` entries with type `location` are usable in `array_locations`; `array_zones` refers to the custom zones. Names must be strings and cannot be substrings of other declared names.

Some example files use a different shape: top-level `locations`, entry type `place`, or `centroid` and dimensions there. Those spellings are not converted by `input_json.py`; use the `places` format above when using this parser.

## Parameters

Each item in `parameters` has a unique `name`, a `type`, and usually a `value`. A missing `value` is stored as `NaN`, but is not useful for most parameter types. When a unit is supplied, numeric values are converted to the generator's base units: seconds, metres/second, and metres.

| `type` accepted by `input_json.py` | Meaning and item-specific fields |
| --- | --- |
| `string` | String `value`. |
| `integer` | Integer `value`. |
| `float` | Float `value` (JSON numeric value with a decimal point is expected). |
| `speed` | Numeric `value`; specify `speed_unit`. |
| `length` | Numeric `value`; specify `length_unit`. |
| `array_locations` | A list of custom location names in `value` (defaults to empty), optional non-negative integer `size`, and required `locs`: `random`, `schools`, or `hospitals`. `random` fills to `size` with generated network points; the other modes select network POIs. An empty `value` with `locs: hospitals` means all available hospitals. |
| `array_zones` | A list of zone names in `value` (defaults to empty), and optional non-negative integer `size`. |
| `array_primitives` | Generic primitive-array parameter. The parser stores its `value`; it performs no additional validation or conversion. |
| `matrix` | Matrix flag/type. The parser sets its internal `value` to true for `travel_time_matrix`; it is not a general-purpose JSON matrix parameter. |
| `graphml` | GraphML flag/type, commonly used with `value: true`. |

An item can also have `time_unit` (`s`, `min`, `h`), `speed_unit` (`mps`, `kmh`, `miph`), or `length_unit` (`m`, `km`, `mi`). Units are converted regardless of the declared type; the matching value must be numeric and non-negative.

The examples also contain parameter type spellings `time`, `list_places`, and `list_zones`. These are **not** in the accepted type list in `input_json.py`; the closest parser types are respectively a numeric parameter with `time_unit`, `array_locations`, and `array_zones`. `time` used as a parameter type should not be confused with `time_unit`. Some old `array_locations`/`array_zones` examples use `list` instead of `value`; the parser reads `value`, not `list`.

## Attributes

Each attribute has a unique `name` and required `type`. The accepted attribute types in `input_json.py` are:

| Type | Meaning |
| --- | --- |
| `integer` | Numeric value converted to an integer when sampled from a distribution or expression. |
| `real` | Numeric/continuous value. |
| `string` | String attribute. |
| `location` | Generated coordinate. The conventional attribute names `origin` and `destination` are used by several expressions. |
| `array_primitives` | Array-valued attribute, commonly used for the bus stops returned by `stops(...)`. |

An attribute is usually generated from `pdf` or computed from `expression`. A `location` attribute is generated as a coordinate. The parser does not enforce that exactly one of `pdf` and `expression` is present. For numeric/time/distance attributes, normally provide one. `subset_zones`, `subset_locations`, and `subset_primitives` refer to parameter names and restrict the generated value to the corresponding configured set. `weights` supplies selection weights for supported uniform/zone/location subsets; the special first value `"randomized_weights"` requests generated weights. The weight count must match the subset size.

Attribute units are optional:

| Field | Accepted units | Stored base unit |
| --- | --- | --- |
| `time_unit` | `s`, `min`, `h` | seconds |
| `speed_unit` | `mps`, `kmh`, `miph` | metres/second |
| `length_unit` | `m`, `km`, `mi` | metres |

The examples also use attribute type spellings `time`, `coordinate`, and `float`; those are not in the parser's accepted attribute type list. Use `integer` with `time_unit` for time quantities, `real` for continuous quantities, and `location` for coordinates.

### Probability distributions (`pdf`)

`pdf` is a list of distribution objects. The implementation reads the **first** object. Distribution names and fields are:

| `pdf[].type` | Required numeric fields | Meaning in the sampler |
| --- | --- | --- |
| `normal` | `loc`, `scale` | Normal distribution; `loc` is the mean and `scale` the standard deviation. |
| `uniform` | `loc`, `scale` | Uniform on `[loc, loc + scale]`; `scale` is the width, not the upper endpoint. |
| `cauchy` | `loc`, `scale` | Cauchy location and scale. |
| `expon` | `loc`, `scale` | Exponential location and scale. |
| `gamma` | `loc`, `scale`, `aux` | Gamma distribution; `aux` is the shape parameter. |
| `gilbrat` | `loc`, `scale` | Gilbrat distribution (uses the available SciPy implementation). |
| `lognorm` | `loc`, `scale` | Log-normal location and scale. |
| `powerlaw` | `loc`, `scale`, `aux` | Power-law distribution; `aux` is its shape parameter. |
| `wald` | `loc`, `scale` | Wald/inverse-Gaussian location and scale. |
| `poisson` | `a` | Poisson parameter used as its rate/mean. |

With `time_unit`, `speed_unit`, or `length_unit`, `loc` and `scale` are converted to the corresponding base unit for `normal` and `uniform`. For other listed distributions, the parser checks numeric values but does not apply that conversion. The parser requires `loc` and `scale` for the distributions that use them and `aux` for `gamma`/`powerlaw`; `poisson` requires `a`.

Some older example attributes use `mean` and `std` in place of `loc` and `scale`. The current attribute parser does not translate those names: use `loc` and `scale` for `normal`. The separate `RequestDistributionTime` helper has its own `mean`/`std` and `min_time`/`max_time` interface; it is not the attribute `pdf` schema described here.

### Expressions and constraints

`expression` is a Python-like arithmetic expression evaluated after its attribute dependencies. References to previously generated attributes and named parameter values are substituted into the expression. It is not a general Python execution environment: its built-in functions are limited to `len`, `set`, `abs`, `float`, `int`, `max`, `min`, `pow`, `round`, `str`, and `isinstance`. The generator also handles the network-specific functions below.

`constraints` is a list of boolean expressions. Attribute and parameter references are substituted; each constraint must evaluate true for the value to be accepted. The generator retries infeasible samples (up to 100 attempts for an attribute; an expression-dependent failure can trigger regeneration of the request).

| Expression function | Returns |
| --- | --- |
| `dtt(origin, destination)` | Estimated direct driving travel time between two generated location attributes, in seconds. |
| `dist_drive(origin, destination)` | Estimated direct driving distance between two generated location attributes, in metres. |
| `walk(origin, destination)` | Estimated walking time between two location attributes, using the generated `walk_speed`, in seconds. |
| `stops(location)` | List of bus-stop indices reachable from the location within `max_walking` at `walk_speed`. It also produces an associated walking-time list for `stops_orgn`/`stops_dest`. |

For `stops(...)`, define the `max_walking` and `walk_speed` attributes. Constraints can use the safe built-ins, for example `len(stops_orgn) > 0`. The runtime's stop handling expects conventional `origin`/`destination` and `stops_orgn`/`stops_dest` attribute names.

### Dynamism, static requests, and CSV output

* `dynamism` sets the target dynamism level for the `time_stamp` attribute. It is interpreted as a percentage (for example `85` or `100`); the parser copies the value without range validation. The generation loop adjusts timestamps until the target level is reached. Use this with a sampled `time_stamp`.
* `static_probability` is recognized only on `time_stamp`, must be a JSON float between 0 and 1, and defaults to `0`. In the expression-based `time_stamp` path, the selected fraction of requests is made static by evaluating the timestamp as zero. The PDF-based timestamp path does not consult it.
* `output_csv` must be a boolean and defaults to `true`. It is stored on the attribute. The current CSV converter does not apply the flag when writing rows (the filtering conditional is commented out), so setting it to `false` does not currently suppress that column.

## Problem types

The JSON `problem` value labels the instance; most of the problem-specific data is described by the attributes you configure. The web UI provides these presets:

* **DARP**: General origin-to-destination requests. Typical attributes include `origin`, `destination`, pickup/drop-off time windows, `direct_travel_time`, and optionally `direct_distance`.
* **ODBRP**: Bus-based origin/destination requests. The example uses `stops_orgn` and `stops_dest` to represent reachable bus stops and includes `max_walking` and `walk_speed`; `travel_time_matrix` can request stop travel times.
* **Patient Transport**: A UI preset for non-urgent medical trips, represented in generated JSON as `"problem": "DARP"`. It uses patient pickup and hospital drop-off attributes; the destination is restricted to a `hospitals` location parameter. It is therefore a DARP variant, not a distinct `problem` value understood by the request generator.

The generator also has lower-level support for labels `ODBRPFL` and `SBRP`. This reference does not assign behavior to those modes beyond the names accepted by `Instance.set_problem_type`.

## Complete examples

The annotations below describe the key groups and non-obvious values in each complete JSON object. Keep the configuration itself valid JSON (comments are not JSON).

### Minimal DARP

`network`, `problem`, `seed`, `replicas`, `requests`, and `instance_filename` identify what to generate. The empty `parameters` list initializes the location registry required by `origin` and `destination`. The time attributes use seconds; `time_stamp` is sampled uniformly across the morning, and the other times are derived from it. `origin` and `destination` are random network coordinates.

```json
{
  "network": "Aachen, Germany",
  "problem": "DARP",
  "seed": 42,
  "replicas": 1,
  "requests": 10,
  "instance_filename": [
    "network",
    "problem",
    "requests"
  ],
  "parameters": [],
  "attributes": [
    {
      "name": "time_stamp",
      "type": "integer",
      "time_unit": "s",
      "pdf": [
        {
          "type": "uniform",
          "loc": 25200,
          "scale": 3600
        }
      ],
      "constraints": [
        "time_stamp >= 0"
      ]
    },
    {
      "name": "earliest_departure",
      "type": "integer",
      "time_unit": "s",
      "expression": "time_stamp"
    },
    {
      "name": "latest_arrival",
      "type": "integer",
      "time_unit": "s",
      "expression": "earliest_departure + 1800"
    },
    {
      "name": "origin",
      "type": "location"
    },
    {
      "name": "destination",
      "type": "location"
    }
  ]
}
```

### ODBRP smoke-test configuration

This is `.github/smoke-test/aachen_odbrp_20.json`. The fixed vehicle speed is specified in km/h and converted internally. The time range is 07:00–08:00; `dynamism: 100` requests fully dynamic timestamps. Direct travel time is constrained by `min_dtt`/`max_dtt`; stop lists are built subject to walking limits. `max_walking` and `walk_speed` are retained in JSON output only as configured here, and the travel-time matrix is requested for `bus_stations`.

```json
{
  "seed": 1,
  "network": "Aachen, Germany",
  "problem": "ODBRP",
  "set_fixed_speed": {
    "vehicle_speed_data": 20,
    "vehicle_speed_data_unit": "kmh"
  },
  "replicas": 1,
  "requests": 20,
  "instance_filename": [
    "network",
    "problem",
    "requests",
    "min_early_departure",
    "max_early_departure",
    "dynamism",
    "urgency",
    "geographic_dispersion"
  ],
  "parameters": [
    {
      "name": "min_early_departure",
      "type": "float",
      "value": 7.0,
      "time_unit": "h"
    },
    {
      "name": "max_early_departure",
      "type": "float",
      "value": 8.0,
      "time_unit": "h"
    },
    {
      "name": "min_dtt",
      "type": "integer",
      "value": 180,
      "length_unit": "m"
    },
    {
      "name": "max_dtt",
      "type": "integer",
      "value": 1000,
      "length_unit": "m"
    },
    {
      "name": "graphml",
      "type": "graphml",
      "value": true
    }
  ],
  "attributes": [
    {
      "name": "time_stamp",
      "type": "integer",
      "time_unit": "s",
      "pdf": [
        {
          "type": "uniform",
          "loc": 25200,
          "scale": 3600
        }
      ],
      "constraints": [
        "time_stamp >= 0",
        "time_stamp >= min_early_departure",
        "time_stamp <= max_early_departure"
      ],
      "dynamism": 100
    },
    {
      "name": "reaction_time",
      "type": "integer",
      "time_unit": "s",
      "pdf": [
        {
          "type": "normal",
          "loc": 1800,
          "scale": 300
        }
      ],
      "constraints": [
        "reaction_time >= 0"
      ]
    },
    {
      "name": "earliest_departure",
      "type": "integer",
      "time_unit": "s",
      "expression": "time_stamp",
      "constraints": [
        "earliest_departure >= 0",
        "earliest_departure >= min_early_departure"
      ]
    },
    {
      "name": "latest_departure",
      "type": "integer",
      "time_unit": "s",
      "expression": "time_stamp + reaction_time"
    },
    {
      "name": "latest_arrival",
      "type": "integer",
      "time_unit": "s",
      "expression": "earliest_departure + direct_travel_time + (reaction_time) + 3600"
    },
    {
      "name": "origin",
      "type": "location"
    },
    {
      "name": "destination",
      "type": "location"
    },
    {
      "name": "stops_orgn",
      "type": "array_primitives",
      "expression": "stops(origin)",
      "constraints": [
        "len(stops_orgn) > 0"
      ]
    },
    {
      "name": "stops_dest",
      "type": "array_primitives",
      "expression": "stops(destination)",
      "constraints": [
        "len(stops_dest) > 0",
        "not (set(stops_orgn) & set(stops_dest))"
      ]
    },
    {
      "name": "max_walking",
      "type": "integer",
      "time_unit": "s",
      "pdf": [
        {
          "type": "uniform",
          "loc": 550,
          "scale": 50
        }
      ],
      "output_csv": false
    },
    {
      "name": "walk_speed",
      "type": "real",
      "speed_unit": "mps",
      "pdf": [
        {
          "type": "uniform",
          "loc": 1.38889,
          "scale": 0
        }
      ],
      "output_csv": false
    },
    {
      "name": "direct_travel_time",
      "type": "integer",
      "time_unit": "s",
      "expression": "dtt(origin,destination)",
      "constraints": [
        "direct_travel_time >= min_dtt",
        "direct_travel_time <= max_dtt"
      ]
    },
    {
      "name": "direct_distance",
      "type": "integer",
      "length_unit": "m",
      "expression": "dist_drive(origin,destination)"
    }
  ],
  "travel_time_matrix": [
    "bus_stations"
  ]
}
```

## Example configurations not fully supported by the current parser

The configuration corpus contains a few stale or alternate schema spellings. They are listed here so their appearance in examples is not mistaken for parser support:

* Top-level `locations`, entry type `place`, parameter types `time`, `list_places`, and `list_zones`, the parameter key `list`, and attribute types `time`, `coordinate`, and `float` occur in basic examples but are not accepted/transformed as such by `input_json.py`. Use `places` with `location`/`zone`, `value` with the parser's `array_locations`/`array_zones`, and attribute types `integer`/`real`/`location`.
* `mean`/`std` occur in one example's normal PDF, but the parser expects `loc`/`scale`.
* `PRICE` occurs as an example `problem` value. Its problem-specific behavior could not be identified in the request-generation code; it is copied through like other labels.

When reproducing one of those examples, check its keys against the accepted forms in this reference rather than assuming every checked-in example is valid for the current parser.
