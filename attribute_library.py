"""
Attribute library for REQreate instance generation
Contains common attribute templates for different problem types
"""

# Common ODBRP attributes
ODBRP_ATTRIBUTES = {
    "time_stamp": {
        "name": "time_stamp",
        "description": "Request arrival time",
        "required": True,
        "template": {
            "name": "time_stamp",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 25200, "scale": 3600}],
            "constraints": ["time_stamp >= 0", "time_stamp >= min_early_departure", "time_stamp <= max_early_departure"],
            "dynamism": 0
        }
    },
    "reaction_time": {
        "name": "reaction_time",
        "description": "Time between request and desired pickup (urgency)",
        "required": True,
        "template": {
            "name": "reaction_time",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 300, "scale": 0}]
        }
    },
    "earliest_departure": {
        "name": "earliest_departure",
        "description": "Earliest departure time from origin stop",
        "required": True,
        "template": {
            "name": "earliest_departure",
            "type": "integer",
            "time_unit": "s",
            "expression": "time_stamp",
            "constraints": ["earliest_departure >= 0", "earliest_departure >= min_early_departure"]
        }
    },
    "latest_departure": {
        "name": "latest_departure",
        "description": "Latest departure time from origin stop",
        "required": True,
        "template": {
            "name": "latest_departure",
            "type": "integer",
            "time_unit": "s",
            "expression": "time_stamp + reaction_time"
        }
    },
    "latest_arrival": {
        "name": "latest_arrival",
        "description": "Latest acceptable arrival time at destination",
        "required": True,
        "template": {
            "name": "latest_arrival",
            "type": "integer",
            "time_unit": "s",
            "expression": "earliest_departure + direct_travel_time + 600"
        }
    },
    "origin": {
        "name": "origin",
        "description": "Origin location (coordinates)",
        "required": True,
        "template": {
            "name": "origin",
            "type": "location"
        }
    },
    "destination": {
        "name": "destination",
        "description": "Destination location (coordinates)",
        "required": True,
        "template": {
            "name": "destination",
            "type": "location"
        }
    },
    "stops_orgn": {
        "name": "stops_orgn",
        "description": "Bus stops near origin",
        "required": True,
        "template": {
            "name": "stops_orgn",
            "type": "array_primitives",
            "expression": "stops(origin)",
            "constraints": ["len(stops_orgn) > 0"]
        }
    },
    "stops_dest": {
        "name": "stops_dest",
        "description": "Bus stops near destination",
        "required": True,
        "template": {
            "name": "stops_dest",
            "type": "array_primitives",
            "expression": "stops(destination)",
            "constraints": ["len(stops_dest) > 0", "not (set(stops_orgn) & set(stops_dest))"]
        }
    },
    "max_walking": {
        "name": "max_walking",
        "description": "Maximum walking distance/time",
        "required": False,
        "template": {
            "name": "max_walking",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 550, "scale": 50}],
            "output_csv": False
        }
    },
    "walk_speed": {
        "name": "walk_speed",
        "description": "Walking speed (m/s)",
        "required": False,
        "template": {
            "name": "walk_speed",
            "type": "real",
            "speed_unit": "mps",
            "pdf": [{"type": "uniform", "loc": 1.38889, "scale": 0}],
            "output_csv": False
        }
    },
    "direct_travel_time": {
        "name": "direct_travel_time",
        "description": "Direct travel time between origin and destination",
        "required": True,
        "template": {
            "name": "direct_travel_time",
            "type": "integer",
            "time_unit": "s",
            "expression": "dtt(origin,destination)"
        }
    },
    "direct_distance": {
        "name": "direct_distance",
        "description": "Direct distance between origin and destination",
        "required": True,
        "template": {
            "name": "direct_distance",
            "type": "integer",
            "length_unit": "m",
            "expression": "dist_drive(origin,destination)"
        }
    },
    "load": {
        "name": "load",
        "description": "Passenger load/demand",
        "required": False,
        "template": {
            "name": "load",
            "type": "integer",
            "pdf": [{"type": "uniform", "loc": 1, "scale": 0}]
        }
    },
    "service_time": {
        "name": "service_time",
        "description": "Time required to service the request (boarding/alighting)",
        "required": False,
        "template": {
            "name": "service_time",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 60, "scale": 30}]
        }
    },
    "max_ride_time": {
        "name": "max_ride_time",
        "description": "Maximum time passenger willing to spend in vehicle",
        "required": False,
        "template": {
            "name": "max_ride_time",
            "type": "integer",
            "time_unit": "s",
            "expression": "direct_travel_time * 2"
        }
    }
}

# Common DARP attributes
DARP_ATTRIBUTES = {
    "time_stamp": {
        "name": "time_stamp",
        "description": "Request arrival time",
        "required": True,
        "template": {
            "name": "time_stamp",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 25200, "scale": 3600}],
            "constraints": ["time_stamp >= 0", "time_stamp >= min_early_departure", "time_stamp <= max_early_departure"],
            "dynamism": 0
        }
    },
    "pickup_from": {
        "name": "pickup_from",
        "description": "Earliest pickup time",
        "required": True,
        "template": {
            "name": "pickup_from",
            "type": "integer",
            "time_unit": "s",
            "expression": "time_stamp"
        }
    },
    "pickup_to": {
        "name": "pickup_to",
        "description": "Latest pickup time",
        "required": True,
        "template": {
            "name": "pickup_to",
            "type": "integer",
            "time_unit": "s",
            "expression": "pickup_from + 300"
        }
    },
    "dropoff_from": {
        "name": "dropoff_from",
        "description": "Earliest dropoff time",
        "required": True,
        "template": {
            "name": "dropoff_from",
            "type": "integer",
            "time_unit": "s",
            "expression": "pickup_to + drivingDuration"
        }
    },
    "dropoff_to": {
        "name": "dropoff_to",
        "description": "Latest dropoff time (with max delay)",
        "required": True,
        "template": {
            "name": "dropoff_to",
            "type": "integer",
            "time_unit": "s",
            "expression": "dropoff_from + 600"
        }
    },
    "origin": {
        "name": "origin",
        "description": "Pickup location (coordinates)",
        "required": True,
        "template": {
            "name": "origin",
            "type": "location"
        }
    },
    "destination": {
        "name": "destination",
        "description": "Dropoff location (coordinates)",
        "required": True,
        "template": {
            "name": "destination",
            "type": "location"
        }
    },
    "drivingDuration": {
        "name": "drivingDuration",
        "description": "Direct driving time",
        "required": True,
        "template": {
            "name": "drivingDuration",
            "type": "integer",
            "time_unit": "s",
            "expression": "dtt(origin,destination)"
        }
    },
    "drivingDistance": {
        "name": "drivingDistance",
        "description": "Direct driving distance",
        "required": True,
        "template": {
            "name": "drivingDistance",
            "type": "integer",
            "length_unit": "m",
            "expression": "dist_drive(origin,destination)"
        }
    },
    "load": {
        "name": "load",
        "description": "Passenger load/demand",
        "required": False,
        "template": {
            "name": "load",
            "type": "integer",
            "pdf": [{"type": "uniform", "loc": 1, "scale": 0}]
        }
    },
    "service_time": {
        "name": "service_time",
        "description": "Time required at pickup/dropoff",
        "required": False,
        "template": {
            "name": "service_time",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 60, "scale": 30}]
        }
    },
    "max_ride_time": {
        "name": "max_ride_time",
        "description": "Maximum time passenger willing to spend in vehicle",
        "required": False,
        "template": {
            "name": "max_ride_time",
            "type": "integer",
            "time_unit": "s",
            "expression": "drivingDuration * 2"
        }
    },
    "max_wait_time": {
        "name": "max_wait_time",
        "description": "Maximum waiting time at pickup",
        "required": False,
        "template": {
            "name": "max_wait_time",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 600, "scale": 300}]
        }
    }
}

# Patient Transport (Hospital DARP) attributes
PATIENT_TRANSPORT_ATTRIBUTES = {
    "time_stamp": {
        "name": "time_stamp",
        "description": "Request arrival/booking time",
        "required": True,
        "template": {
            "name": "time_stamp",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 25200, "scale": 3600}],
            "constraints": ["time_stamp >= 0", "time_stamp >= min_early_departure", "time_stamp <= max_early_departure"],
            "dynamism": 0
        }
    },
    "pickup_from": {
        "name": "pickup_from",
        "description": "Earliest pickup time (time window start)",
        "required": True,
        "template": {
            "name": "pickup_from",
            "type": "integer",
            "time_unit": "s",
            "expression": "time_stamp"
        }
    },
    "pickup_to": {
        "name": "pickup_to",
        "description": "Latest pickup time (time window end)",
        "required": True,
        "template": {
            "name": "pickup_to",
            "type": "integer",
            "time_unit": "s",
            "expression": "pickup_from + 900"
        }
    },
    "dropoff_from": {
        "name": "dropoff_from",
        "description": "Earliest dropoff time at medical facility",
        "required": True,
        "template": {
            "name": "dropoff_from",
            "type": "integer",
            "time_unit": "s",
            "expression": "pickup_to + drivingDuration"
        }
    },
    "dropoff_to": {
        "name": "dropoff_to",
        "description": "Latest dropoff time (appointment deadline)",
        "required": True,
        "template": {
            "name": "dropoff_to",
            "type": "integer",
            "time_unit": "s",
            "expression": "dropoff_from + 900"
        }
    },
    "origin": {
        "name": "origin",
        "description": "Pickup location (patient home/residence)",
        "required": True,
        "template": {
            "name": "origin",
            "type": "location"
        }
    },
    "destination": {
        "name": "destination",
        "description": "Dropoff location (hospital/medical facility)",
        "required": True,
        "template": {
            "name": "destination",
            "type": "location",
            "subset_locations": "hospitals"
        }
    },
    "drivingDuration": {
        "name": "drivingDuration",
        "description": "Direct driving time (minimal ride time)",
        "required": True,
        "template": {
            "name": "drivingDuration",
            "type": "integer",
            "time_unit": "s",
            "expression": "dtt(origin,destination)"
        }
    },
    "drivingDistance": {
        "name": "drivingDistance",
        "description": "Direct driving distance",
        "required": True,
        "template": {
            "name": "drivingDistance",
            "type": "integer",
            "length_unit": "m",
            "expression": "dist_drive(origin,destination)"
        }
    },
    "load": {
        "name": "load",
        "description": "Number of patients/users to transport",
        "required": False,
        "template": {
            "name": "load",
            "type": "integer",
            "pdf": [{"type": "uniform", "loc": 1, "scale": 0}]
        }
    },
    "service_time": {
        "name": "service_time",
        "description": "Service duration at pickup/dropoff (loading/unloading/assistance)",
        "required": False,
        "template": {
            "name": "service_time",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 180, "scale": 120}]
        }
    },
    "max_ride_time": {
        "name": "max_ride_time",
        "description": "Maximal ride time constraint (single trip)",
        "required": False,
        "template": {
            "name": "max_ride_time",
            "type": "integer",
            "time_unit": "s",
            "expression": "drivingDuration * 1.5"
        }
    },
    "max_daily_ride_time": {
        "name": "max_daily_ride_time",
        "description": "Maximal daily ride time (round-trip constraint)",
        "required": False,
        "template": {
            "name": "max_daily_ride_time",
            "type": "integer",
            "time_unit": "s",
            "pdf": [{"type": "uniform", "loc": 7200, "scale": 1800}]
        }
    },
    "excess_ride_time": {
        "name": "excess_ride_time",
        "description": "Allowed extra time beyond minimal ride time",
        "required": False,
        "template": {
            "name": "excess_ride_time",
            "type": "integer",
            "time_unit": "s",
            "expression": "max_ride_time - drivingDuration"
        }
    }
}

def get_attributes_for_problem(problem_type):
    """Get default attributes for a problem type"""
    if problem_type == "ODBRP":
        return ODBRP_ATTRIBUTES
    elif problem_type == "DARP":
        return DARP_ATTRIBUTES
    elif problem_type == "Patient Transport":
        return PATIENT_TRANSPORT_ATTRIBUTES
    else:
        return {}

def get_required_attributes(problem_type):
    """Get list of required attribute names"""
    attrs = get_attributes_for_problem(problem_type)
    return [name for name, info in attrs.items() if info.get("required", False)]

def build_attributes_list(problem_type, selected_attrs, custom_values=None):
    """
    Build the attributes list for JSON config
    
    Args:
        problem_type: "ODBRP" or "DARP"
        selected_attrs: List of attribute names to include
        custom_values: Dict of custom values to override defaults
    """
    attrs = get_attributes_for_problem(problem_type)
    attributes_list = []
    
    for attr_name in selected_attrs:
        if attr_name in attrs:
            attr = attrs[attr_name]["template"].copy()
            
            # Apply custom values if provided
            if custom_values and attr_name in custom_values:
                for key, value in custom_values[attr_name].items():
                    attr[key] = value
            
            attributes_list.append(attr)
    
    return attributes_list
