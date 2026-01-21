import argparse
import boto3
import datetime
import json
import pathlib
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Mapping from our trace names to official AWS instance types
INSTANCE_TYPE_MAP = {
    "k80_1": "p2.xlarge",
    "k80_8": "p2.8xlarge",
    "v100_1": "p3.2xlarge",
    "v100_8": "p3.16xlarge",
}

# The regions we are interested in, based on trace data
AVAILABILITY_ZONES = [
    "us-west-2a",
    "us-west-2b",
    "us-west-2c",
    "us-east-1a",
    "us-east-1c",
    "us-east-1d",
    "us-east-1f",
    "us-east-2a",
    "us-east-2b",
]

REGION_TO_LOCATION = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-2": "US West (Oregon)",
}


def fetch_on_demand_price(instance_type: str, region: str) -> Optional[float]:
    print(f"Fetching On-Demand price for {instance_type} in {region}...")
    pricing_client = boto3.client("pricing", region_name="us-east-1")
    location = REGION_TO_LOCATION.get(region)
    if not location:
        print(
            f"  -> WARNING: Region {region} not mapped to a location. Cannot get On-Demand price."
        )
        return None

    paginator = pricing_client.get_paginator("get_products")
    pages = paginator.paginate(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "location", "Value": location},
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {
                "Type": "TERM_MATCH",
                "Field": "tenancy",
                "Value": "Shared",
            },  # Shared tenancy for standard On-Demand
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {
                "Type": "TERM_MATCH",
                "Field": "preInstalledSw",
                "Value": "NA",
            },  # No pre-installed software
        ],
    )
    for page in pages:
        for price_data in page["PriceList"]:
            price_list = json.loads(price_data)
            on_demand_terms = price_list.get("terms", {}).get("OnDemand", {})
            if not on_demand_terms:
                continue

            term_code = list(on_demand_terms.keys())[0]
            price_dimensions = on_demand_terms[term_code].get("priceDimensions", {})

            dim_code = list(price_dimensions.keys())[0]
            price_info = price_dimensions[dim_code]

            price_str = price_info.get("pricePerUnit", {}).get("USD")
            if price_str:
                print(f"  -> Found On-Demand price: ${price_str}")
                return float(price_str)

    return None


def fetch_price_history(
    client, instance_type: str, availability_zone: str, days: int
) -> List[Dict]:
    """Fetches spot price history for a given instance type and AZ."""
    print(
        f"Fetching prices for {instance_type} in {availability_zone} for the last {days} days..."
    )
    start_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=days
    )
    paginator = client.get_paginator("describe_spot_price_history")
    pages = paginator.paginate(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        AvailabilityZone=availability_zone,
        StartTime=start_time,
    )
    price_history = []
    try:
        for page in pages:
            for price_point in page["SpotPriceHistory"]:
                price_history.append(
                    {
                        "Timestamp": price_point["Timestamp"].isoformat(),
                        "SpotPrice": price_point["SpotPrice"],
                    }
                )
    except client.exceptions.ClientError as e:
        print(
            f"  -> WARNING: AWS API error for {availability_zone}: {e}. This might be due to permissions or an invalid zone. Skipping."
        )
        return []

    price_history.sort(key=lambda x: x["Timestamp"])
    return price_history


def plot_price_comparison(data_dir: pathlib.Path, target_instance_short_name: str):
    """Plots a comparison of spot prices for a given instance across all fetched AZs."""
    print(f"\n🎨 Generating plot for {target_instance_short_name}...")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))

    price_files = sorted(data_dir.glob(f"*_{target_instance_short_name}_prices.json"))

    if not price_files:
        print(
            f"  -> No data files found for {target_instance_short_name}. Skipping plot."
        )
        plt.close(fig)
        return

    aws_instance_type = INSTANCE_TYPE_MAP.get(
        target_instance_short_name, target_instance_short_name
    )
    first_az = price_files[0].name.split("_")[0]
    region = first_az[:-1]  # e.g., 'us-west-2a' -> 'us-west-2'
    on_demand_price = fetch_on_demand_price(aws_instance_type, region)

    colors = plt.cm.viridis(np.linspace(0, 1, len(price_files)))

    for i, file_path in enumerate(price_files):
        az_name = file_path.name.split("_")[0]

        try:
            df = pd.read_json(file_path)
            if df.empty:
                continue
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["SpotPrice"] = pd.to_numeric(df["SpotPrice"])
            ax.plot(
                df["Timestamp"],
                df["SpotPrice"],
                label=az_name,
                color=colors[i],
                linewidth=2,
            )
        except Exception as e:
            print(f"  -> Could not process or plot file {file_path}: {e}")

    if on_demand_price is not None:
        ax.axhline(
            y=on_demand_price,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=f"On-Demand: ${on_demand_price:.4f}",
        )

    ax.set_title(
        f"Historical Spot Price Comparison: {target_instance_short_name} ({aws_instance_type})",
        fontsize=22,
        pad=20,
        weight="bold",
    )
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Spot Price ($ / hour)", fontsize=16)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:.3f}"))
    fig.autofmt_xdate(rotation=45)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.legend(
        title="Availability Zone / Type",
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    ax.grid(True, which="major", linestyle="--", linewidth=0.7)

    if on_demand_price:
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], max(current_ylim[1], on_demand_price * 1.1))

    output_filename = f"spot_price_comparison_{target_instance_short_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"  -> ✅ Plot saved successfully to {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch AWS Spot Price history and save it to JSON files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/spot_prices",
        help="Directory to save pricing data.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of past days to fetch history for (max 90).",
    )
    args = parser.parse_args()

    output_dir_path = pathlib.Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(
        "This script requires AWS credentials to be configured (e.g., via `aws configure`)."
    )

    clients: Dict[str, "boto3.client"] = {}

    for zone in AVAILABILITY_ZONES:
        region = zone[:-1]
        if region not in clients:
            print(f"\nCreating boto3 client for region: {region}")
            clients[region] = boto3.client("ec2", region_name=region)
        client = clients[region]

        for short_name, aws_instance_type in INSTANCE_TYPE_MAP.items():
            price_data = fetch_price_history(client, aws_instance_type, zone, args.days)
            if not price_data:
                print(
                    f"  -> No price data found for {aws_instance_type} in {zone}. Skipping."
                )
                continue

            output_filename = f"{zone}_{short_name}_prices.json"
            output_path = output_dir_path / output_filename
            with open(output_path, "w") as f:
                json.dump(price_data, f, indent=2)
            print(
                f"  -> Successfully saved {len(price_data)} price points to {output_path}"
            )

    print("\n----------------------------------------")
    print("All data fetched. Now generating plots...")

    for short_name in INSTANCE_TYPE_MAP.keys():
        plot_price_comparison(output_dir_path, short_name)


if __name__ == "__main__":
    main()
