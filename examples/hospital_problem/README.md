# Patient Transport Problem

This directory contains examples for generating non-urgent patient transportation instances using the REQreate framework.

## Problem Description

The **Patient Transport Problem** is a specialized variant of the Dial-a-Ride Problem (DARP) designed for medical transportation services. It includes specific characteristics for transporting patients to/from healthcare facilities:

### Key Features

1. **Pickup and Delivery Nodes**
   - **Pickup**: Patient homes or residences (random locations)
   - **Delivery**: Medical facilities (hospitals, clinics) from OpenStreetMap

2. **Time Windows**
   - Flexible pickup windows: `[pickup_from, pickup_to]` (e.g., 15-60 minutes)
   - Appointment deadline: `[dropoff_from, dropoff_to]`
   - Configurable based on patient type (urgent vs. routine appointments)

3. **Ride Time Constraints**
   - **Max Ride Time**: 150% of minimal (direct) ride time
   - **Max Daily Ride Time**: Constraint for round-trip journeys (2-3 hours)
   - **Excess Ride Time**: Additional time beyond direct route
   - **Minimal Ride Time**: Direct travel time between locations

4. **Service Duration**
   - Longer service times (3-5 minutes) to account for:
     - Patient boarding/alighting assistance
     - Medical equipment handling
     - Wheelchair loading/unloading

5. **User Quantity (Load)**
   - Number of patients per request
   - Affects vehicle capacity planning

## Attributes

### Required Attributes
- `time_stamp`: Request booking time
- `pickup_from`, `pickup_to`: Pickup time window
- `dropoff_from`, `dropoff_to`: Dropoff time window (appointment)
- `origin`: Patient location (home)
- `destination`: Hospital location (constrained to hospitals)
- `drivingDuration`: Minimal ride time
- `drivingDistance`: Direct distance

### Optional Attributes
- `load`: Number of patients (default: 1)
- `service_time`: Assistance time (3-5 min)
- `max_ride_time`: 150% of direct time
- `max_daily_ride_time`: Round-trip constraint (2-3 hours)
- `excess_ride_time`: Allowed detour time

## Usage

### Prerequisites

1. **Generate network with hospitals**:
   ```bash
   python add_hospitals_to_existing_network.py "City, Country"
   ```

2. **Or create a new network** (automatically includes hospitals):
   - Use the web interface: "Create New Instance"
   - Hospitals are retrieved from OpenStreetMap

### Web Interface

1. Go to "Create New Instance"
2. Select **"Patient Transport"** as Problem Type
3. Configure:
   - Number of requests (patients)
   - Time horizon (operating hours)
   - Attributes (required + optional)
4. Generate instance

### Command Line

```bash
python -m REQreate.REQreate examples/hospital_problem/patient_transport_example.json
```

## Example Configuration

See `patient_transport_example.json` for a complete configuration.

Key differences from standard DARP:
```json
{
  "problem": "DARP",
  "attributes": [
    {
      "name": "destination",
      "type": "location",
      "subset_locations": "hospitals"  // Constrains to hospital locations
    },
    {
      "name": "max_ride_time",
      "type": "integer",
      "time_unit": "s",
      "expression": "drivingDuration * 1.5"  // 150% of direct time
    },
    {
      "name": "service_time",
      "type": "integer",
      "time_unit": "s",
      "pdf": [{"type": "uniform", "loc": 180, "scale": 120}]  // 3-5 minutes
    }
  ]
}
```

## Real-World Applications

- Hospital shuttle services
- Non-emergency medical transportation (NEMT)
- Dialysis patient transport
- Chemotherapy appointment transport
- Rehabilitation center shuttles
- Senior care facility transport

## Notes

- Uses DARP solver with hospital-specific constraints
- Destinations are automatically constrained to retrieved hospital locations
- Service times are longer than standard ride-sharing (patient assistance)
- Time windows can be flexible for "chronic" patients with routine appointments
- Round-trip constraints help limit total patient travel burden
