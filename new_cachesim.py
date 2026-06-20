"""Cache simulator for architectural hardware performance evaluation.

This module orchestrates the execution pipeline of a configurable CPU cache
    simulator. It processes command-line configurations, validates architectural
    parameters (e.g., power-of-two constraints), streams and parses memory trace
    addresses, runs cache lookup and replacement algorithms, and exports final
    metrics to stdout, JSON, or CSV file formats.

    Supported placement policies include Direct-Mapped, Fully Associative, and
    Set-Associative configurations using Least Recently Used (LRU) and First-In,
    First-Out (FIFO) replacement algorithms.

Example:
    $ python cache_sim.py --size 32768 --assoc 4 --block 64 trace.txt

Attributes:
    None
"""

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class ReplacementPolicy(Enum):  # Enum for replacement policies
    """Enumeration of supported cache block eviction strategies.

    Attributes:
        LRU: Least Recently Used - evicts the block that has not been accessed for the longest time.
        FIFO: First-In-First-Out - evicts the block that was inserted earliest into the set.
        RANDOM: Random - evicts a randomly selected block from the set.
    """

    LRU = "LRU"
    FIFO = "FIFO"
    RANDOM = "random"

    @classmethod
    def _missing_(
        cls, value: object
    ) -> "ReplacementPolicy | None":  # Single underscore for Enums
        """Handles lookup attempts for invalid or lowercase policy strings.

        Provides case-insensitive lookup flexibility when instantiating the
        enum from configuration strings or user input.

        Args:
            value: The lookup value that failed the standard, exact-match lookup.

        Returns:
            The matching ReplacementPolicy enum member if a case-insensitive
            match is found, otherwise None.
        """
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


@dataclass
class CacheConfig:  # Holds simulator settings in one place
    """Configuration parameters for a cache simulation.

    Attributes:
        block_size: Cache block size in bytes.
        num_blocks: Total number of cache blocks.
        associativity: Number of ways per set.
        replacement_policy: Cache replacement policy.
        datafile: Path to the memory trace file.
    """

    block_size: int
    num_blocks: int
    associativity: int
    replacement_policy: ReplacementPolicy
    datafile: Path

    @property
    def num_sets(self) -> int:
        return self.num_blocks // self.associativity

    @property
    def offset_bits(self) -> int:
        return self.block_size.bit_length() - 1  # Number of bits for block offset

    @property
    def index_bits(self) -> int:
        return self.num_sets.bit_length() - 1  # Number of bits for set index


@dataclass
class CacheBlock:  # Represents a single cache line/block
    """Represents a single cache line or block within a set.

    Attributes:
        tag: The unique memory address identifier stored in this block.
        last_used: A cycle timestamp tracking the most recent access (for LRU).
        insertion_time: A cycle timestamp tracking when the block arrived (for FIFO).
    """

    tag: int
    last_used: int = 0  # Used by LRU
    insertion_time: int = 0  # Used by FIFO


@dataclass
class CacheSet:  # Represents a set in the cache
    """Represents a set containing multiple cache blocks.

    Attributes:
        blocks: A list of cache blocks in this set.
        hits: The number of cache hits in this set.
        misses: The number of cache misses in this set.
    """

    blocks: list[CacheBlock] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (
            (self.hits / self.total_accesses) * 100 if self.total_accesses > 0 else 0.0
        )

    @property
    def miss_rate(self) -> float:
        return (
            (self.misses / self.total_accesses) * 100
            if self.total_accesses > 0
            else 0.0
        )


@dataclass
class CacheStats:
    """Aggregates overall performance metrics across the entire simulator execution.

    Attributes:
        total_hits: Consolidated number of cache hits across all sets.
        total_misses: Consolidated number of cache misses across all sets.
        cache: The global architectural matrix containing all active CacheSets.
    """

    total_hits: int
    total_misses: int
    cache: list[CacheSet]

    @property
    def total_accesses(self) -> int:
        return self.total_hits + self.total_misses

    @property
    def hit_rate(self) -> float:
        return (
            (self.total_hits / self.total_accesses) * 100
            if self.total_accesses > 0
            else 0.0
        )

    @property
    def miss_rate(self) -> float:
        return (
            (self.total_misses / self.total_accesses) * 100
            if self.total_accesses > 0
            else 0.0
        )


@dataclass
class SweepResult:
    """Stores the results of a single cache sweep configuration.

    This dataclass aggregates the simulation metrics and calculated performance
    rates for a specific cache configuration evaluated during a sweep.

    Attributes:
        block_size: The size of each cache block in bytes.
        num_blocks: The total number of blocks in the cache.
        associativity: The associativity of the cache (e.g., 1 for direct-mapped).
        replacement_policy: The cache replacement policy utilized during simulation.
        total_accesses: The aggregate number of cache accesses.
        total_hits: The total number of successful cache hits.
        total_misses: The total number of cache misses.
        hit_rate: The ratio of cache hits to total accesses.
        miss_rate: The ratio of cache misses to total accesses.
    """

    block_size: int
    num_blocks: int
    associativity: int
    replacement_policy: str

    total_accesses: int
    total_hits: int
    total_misses: int

    hit_rate: float
    miss_rate: float


def is_power_of_two(value: int) -> bool:
    """Checks if a number is an integral power of two using bitwise operations.

    Args:
        value: The integer number to check.

    Returns:
        True if the number is a power of two, False otherwise.
    """
    # A power of two has exactly one bit set to '1' in its binary representation.
    # Subtracting 1 flips all bits up to that '1', causing bitwise AND to yield 0 for powers of two.
    return value > 0 and (value & (value - 1)) == 0


def validate_config(config: CacheConfig) -> None:
    """Validates the cache architecture configuration parameters and input file.

    Ensures that all parameters adhere to hardware simulation limits, geometric
    constraints, and that the trace file exists.

    Args:
      config: A CacheConfig object containing the cache parameters and datafile path.

    Raises:
      ValueError: If any of the cache parameters are invalid (e.g., non-positive block size, associativity not a power of two, etc.)
      FileNotFoundError: If the specified datafile does not exist.
    """
    if config.block_size <= 0:
        raise ValueError("Block size must be a positive integer.")

    if config.num_blocks <= 0:
        raise ValueError("Number of blocks must be a positive integer.")

    if config.num_blocks % config.associativity != 0:
        raise ValueError("Number of blocks must be a multiple of associativity.")

    if not is_power_of_two(config.associativity):
        raise ValueError("Associativity must be a power of two.")

    if config.associativity > config.num_blocks:
        raise ValueError("Associativity cannot be greater than the number of blocks.")

    if not is_power_of_two(config.block_size):
        raise ValueError("Block size must be a power of two.")

    if not is_power_of_two(config.num_sets):
        raise ValueError(
            "Number of sets (num_blocks / associativity) must be a power of two."
        )

    if not Path(config.datafile).is_file():
        raise FileNotFoundError(f"Datafile '{config.datafile}' not found.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the hardware cache simulator.

    Defines and processes input constraints for cache geometries (block size,
    block count, associativity), eviction choices, and the memory trace file.
    Also allows optional output formats (CSV, JSON).

    Returns:
        An argparse.Namespace object containing the parsed command-line parameters
        mapped to their respective configuration attributes.
    """
    parser = argparse.ArgumentParser(description="Cache Simulator")
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="Block size in bytes (must be > 0 and a power of two)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=256,
        help="Number of blocks in the cache (must be > 0)",
    )
    parser.add_argument(
        "--associativity",
        type=int,
        default=1,
        help="Cache associativity (1 = direct-mapped, 2 = 2-way, 4 = 4-way, etc. Must be a power of two and less than or equal to num_blocks.)",
    )
    parser.add_argument(
        "--replacement_policy",
        choices=[p.value for p in ReplacementPolicy],
        default=ReplacementPolicy.LRU.value,
        help="Replacement policy (LRU, FIFO, or random)",
    )
    parser.add_argument(
        "--datafile",
        type=str,
        required=True,
        help="File with memory addresses (one address per line)",
    )
    parser.add_argument(
        "--json_output", type=str, help="Optional JSON file to save results"
    )
    parser.add_argument(
        "--csv_output", type=str, help="Optional CSV file to save results"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a parameter sweep instead of a single simulation.",
    )
    parser.add_argument(
        "--sweep_block_sizes",
        nargs="+",
        type=int,
        help="Block sizes (in bytes) to test during sweep mode.",
    )
    parser.add_argument(
        "--sweep_num_blocks",
        nargs="+",
        type=int,
        help="Cache sizes (in blocks) to test during sweep mode.",
    )
    parser.add_argument(
        "--sweep_associativities",
        nargs="+",
        type=int,
        help="Associativities to test during sweep mode.",
    )
    parser.add_argument(
        "--sweep_policies",
        nargs="+",
        type=str,
        help="Replacement policies to test during sweep mode.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> CacheConfig:
    """Construct a validated CacheConfig object from parsed command-line arguments.

    Extracts raw specifications and path boundaries from the CLI namespace and
    initializes them into structured simulator parameters.

    Args:
      args: The parsed command-line argument namespace containing attributes for
            block_size, num_blocks, associativity, replacement_policy, and datafile.

    Returns:
      A structured CacheConfig object containing the parsed configuration ready for validation.
    """
    return CacheConfig(
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        associativity=args.associativity,
        replacement_policy=ReplacementPolicy(args.replacement_policy),
        datafile=Path(args.datafile),
    )


def load_addresses(filename: Path) -> list[int]:
    """Stream and parse hexadecimal memory addresses from a trace data file.

    Reads a text file line-by-line, skips empty spacing, and converts raw hex
    strings into base-10 integers ready for simulator bit-masking operations.

    Args:
      filename: A Path object pointing to the file containing memory addresses.

    Raises:
      ValueError: If a non-empty line cannot be parsed as a valid hexadecimal
                  number, indicating a malformed trace entry or corrupt/invalid data.

    Returns:
      A list of parsed memory addresses represented as integers.
    """
    addresses = []

    with filename.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                addresses.append(int(line, 16))  # Convert hex string to integer
            except ValueError:
                raise ValueError(
                    f"Invalid address '{line}' on line {line_number} in datafile. Must be a valid hexadecimal number."
                )

    return addresses


def decode_address(address: int, config: CacheConfig) -> tuple[int, int]:
    """Extract the tag and set index fields from a raw memory address.

    Uses bitwise right-shifts and generated bitmasks based on the cache's
    configured offset and index bit widths to isolate specific fields.

    Args:
      address: The raw integer memory address to decode.
      config: The structured CacheConfig object providing geometric bit widths.

    Returns:
      A tuple containing:
        tag: The unique identifier tag for the memory block.
        set_index: The target cache set array index where the block resides.
    """
    offset_bits = config.offset_bits
    index_bits = config.index_bits
    set_index = (address >> offset_bits) & ((1 << index_bits) - 1)  # Extract set index
    tag = address >> (offset_bits + index_bits)  # Extract tag

    return tag, set_index


def simulate_cache(addresses: list[int], config: CacheConfig) -> CacheStats:
    """Execute the functional simulation of the memory cache architecture.

    Orchestrates the global simulation state by streaming access addresses, parsing
    tags, tracking state tracking histories, and running hit, miss, and eviction
    routines based on the configuration's architectural policies (LRU, FIFO, RANDOM).

    Args:
      addresses: A sequence of parsed memory addresses represented as base-10 integers.
      config: The structured configuration container holding system geometries and policies.

    Returns:
      An aggregated CacheStats object holding hit counts, miss counts,
      and final structural cache states.
    """
    num_sets = config.num_sets
    cache = [CacheSet() for _ in range(num_sets)]
    total_hits, total_misses = 0, 0
    time = 0  # Global time counter for LRU/FIFO
    for address in addresses:
        time += 1
        tag, set_index = decode_address(address, config)
        cache_set = cache[set_index]
        hit_block = None

        # Search for matching tag
        for block in cache_set.blocks:
            if block.tag == tag:
                hit_block = block
                break

        # Cache hit
        if (
            hit_block is not None
        ):  # Explicit check for None because 0 is a valid block despite being falsy
            total_hits += 1
            cache_set.hits += 1
            if config.replacement_policy == ReplacementPolicy.LRU:
                hit_block.last_used = time  # Update usage time for LRU

        # Cache miss
        else:
            total_misses += 1
            cache_set.misses += 1
            new_block = CacheBlock(tag=tag, last_used=time, insertion_time=time)

            # Space available in set
            if len(cache_set.blocks) < config.associativity:
                cache_set.blocks.append(new_block)

            # Need to evict a block
            else:
                if config.replacement_policy == ReplacementPolicy.LRU:
                    victim_index = min(
                        range(len(cache_set.blocks)),
                        key=lambda idx: cache_set.blocks[idx].last_used,
                    )  # Find least recently used block

                elif config.replacement_policy == ReplacementPolicy.FIFO:
                    victim_index = min(
                        range(len(cache_set.blocks)),
                        key=lambda idx: cache_set.blocks[idx].insertion_time,
                    )  # Find first inserted block

                elif config.replacement_policy == ReplacementPolicy.RANDOM:
                    victim_index = random.randrange(
                        len(cache_set.blocks)
                    )  # Randomly select a block to evict

                cache_set.blocks[victim_index] = (
                    new_block  # Replace victim block with new block
                )

    return CacheStats(total_hits=total_hits, total_misses=total_misses, cache=cache)


def run_sweep(addresses: list[int], args: argparse.Namespace) -> list[SweepResult]:
    """Runs a parameter sweep simulation over various cache configurations.

    Iterates through combinations of block sizes, number of blocks,
    associativities, and replacement policies specified in the arguments,
    simulating cache behavior for each valid configuration.

    Args:
        addresses: A list of memory addresses to simulate.
        args: Parsed command-line arguments containing sweep parameters
            (e.g., sweep_block_sizes, sweep_num_blocks,
            sweep_associativities, sweep_policies) and default
            simulation settings.

    Returns:
        A list of results from each valid cache simulation,
        including configuration details and performance metrics.
    """
    results: list[SweepResult] = []

    block_sizes = args.sweep_block_sizes or [args.block_size]
    num_blocks_list = args.sweep_num_blocks or [args.num_blocks]
    associativities = args.sweep_associativities or [args.associativity]

    policies: list[ReplacementPolicy] = (
        [ReplacementPolicy(policy) for policy in args.sweep_policies]
        if args.sweep_policies
        else [ReplacementPolicy(args.replacement_policy)]
    )

    configurations = list(
        product(
            block_sizes,
            num_blocks_list,
            associativities,
            policies,
        )
    )

    for block_size, num_blocks, associativity, policy in tqdm(
        configurations,
        desc="Running sweep",
        unit="config",
    ):
        config = CacheConfig(
            block_size=block_size,
            num_blocks=num_blocks,
            associativity=associativity,
            replacement_policy=policy,
            datafile=Path(args.datafile),
        )

        try:
            validate_config(config)
        except (ValueError, FileNotFoundError):
            continue

        stats = simulate_cache(addresses, config)

        results.append(
            SweepResult(
                block_size=config.block_size,
                num_blocks=config.num_blocks,
                associativity=config.associativity,
                replacement_policy=config.replacement_policy.value,
                total_accesses=stats.total_accesses,
                total_hits=stats.total_hits,
                total_misses=stats.total_misses,
                hit_rate=stats.hit_rate,
                miss_rate=stats.miss_rate,
            )
        )

    return results


def print_sweep_results(results: list[SweepResult]) -> None:
    """Prints a formatted summary of cache simulation sweep results.

    Analyzes a collection of simulation results to identify performance extremes
    and displays a tabular breakdown of the top 10 configurations based on their
    cache hit rate.

    Args:
        results: A list of SweepResult objects containing the performance data
            for various cache configurations.

    Notes:
        - If the results list is empty, a failure message is printed and the
          function exits early to avoid math/comparison errors.
        - Performance rankings ('best', 'worst', and 'top 10') are evaluated
          strictly by the `hit_rate` attribute of the SweepResult.
    """
    if not results:
        print("No valid sweep results generated.")
        return

    print("\nSweep Summary")
    print("-------------")
    print(f"Configurations Tested : {len(results)}")

    best = max(results, key=lambda r: r.hit_rate)
    worst = min(results, key=lambda r: r.hit_rate)

    print(f"Best Hit Rate          : {best.hit_rate:.2f}%")
    print(f"Worst Hit Rate         : {worst.hit_rate:.2f}%")

    print("\nTop 10 Configurations")
    print("---------------------")
    print("Block Size | Blocks | Assoc | Policy | Hit Rate | Miss Rate")

    top_results = sorted(results, key=lambda r: r.hit_rate, reverse=True)[:10]

    for result in top_results:
        print(
            f"{result.block_size:10d} | "
            f"{result.num_blocks:6d} | "
            f"{result.associativity:5d} | "
            f"{result.replacement_policy:6s} | "
            f"{result.hit_rate:8.2f}% | "
            f"{result.miss_rate:9.2f}%"
        )


def print_statistics(stats: CacheStats, config: CacheConfig) -> None:
    """Print the final aggregated performance metrics for a single simulation to the console.

    Outputs structured rows showing total memory access attempts, absolute hit and
    miss counts, and calculated percentage efficiency rates rounded to two decimal places.

    Args:
      stats: The collection containing overall simulator performance metrics.
      config: A CacheConfig ovject containing the physical layout parameters of the
        simulated cache hardware (e.g., size, layout, replacement policy)
    """
    print("\nConfiguration")
    print("-------------")
    print(f"Block Size        : {config.block_size}")
    print(f"Number of Blocks  : {config.num_blocks}")
    print(f"Associativity     : {config.associativity}")
    print(f"Number of Sets    : {config.num_sets}")
    print(f"Replacement Policy: {config.replacement_policy.value}")

    print("\nCache Statistics")
    print("----------------")
    print(f"Accesses : {stats.total_accesses}")
    print(f"Hits     : {stats.total_hits}")
    print(f"Misses   : {stats.total_misses}")
    print(f"Hit Rate : {stats.hit_rate:.2f}%")
    print(f"Miss Rate: {stats.miss_rate:.2f}%")


def save_json(stats: CacheStats, config: CacheConfig, filename: Path) -> None:
    """Serialize the simulator configuration and performance metrics to a JSON file.

    Generates a structured dictionary containing cache geometry, and a detailed breakdown
    of per-set hit/miss ratios, then writes the output to a JSON file.

    Args:
      stats: The collection containing overall simulator performance metrics.
      config: The structured configuration container holding system geometries.
      filename: A Path object specifying the destination target JSON file.
    """
    output = {
        "configuration": {
            "block_size": config.block_size,
            "num_blocks": config.num_blocks,
            "associativity": config.associativity,
            "num_sets": config.num_sets,
            "replacement_policy": config.replacement_policy.value,
        },
        "statistics": {
            "total_accesses": stats.total_accesses,
            "total_hits": stats.total_hits,
            "total_misses": stats.total_misses,
            "hit_rate": stats.hit_rate,
            "miss_rate": stats.miss_rate,
        },
        "sets": [
            {
                "set_index": i,
                "hits": cache_set.hits,
                "misses": cache_set.misses,
                "hit_rate": cache_set.hit_rate,
                "miss_rate": cache_set.miss_rate,
            }
            for i, cache_set in enumerate(stats.cache)
        ],
    }

    with filename.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def save_csv(stats: CacheStats, config: CacheConfig, filename: Path) -> None:
    """Export the simulator configuration and performance metrics to a CSV file.

    Writes a structured, human-readable tabular file partitioned into distinct
    sections covering global system geometry, aggregated simulation performance
    ratios, and a detailed row-by-row matrix of individual cache set histories.

    Args:
      stats: The collection containing overall simulator performance metrics.
      config: The structured configuration container holding system geometries.
      filename: A Path object specifying the destination target CSV file.
    """
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["[Configuration]"])
        writer.writerow(["block_size", config.block_size])
        writer.writerow(["num_blocks", config.num_blocks])
        writer.writerow(["associativity", config.associativity])
        writer.writerow(["num_sets", config.num_sets])
        writer.writerow(["replacement_policy", config.replacement_policy.value])

        writer.writerow([])

        writer.writerow(["[Statistics]"])
        writer.writerow(["total_accesses", stats.total_accesses])
        writer.writerow(["total_hits", stats.total_hits])
        writer.writerow(["total_misses", stats.total_misses])
        writer.writerow(["hit_rate", stats.hit_rate])
        writer.writerow(["miss_rate", stats.miss_rate])

        writer.writerow([])

        writer.writerow(["[Sets]"])
        writer.writerow(["set_index", "hits", "misses", "hit_rate", "miss_rate"])

        for i, cache_set in enumerate(stats.cache):
            writer.writerow(
                [
                    i,
                    cache_set.hits,
                    cache_set.misses,
                    cache_set.hit_rate,
                    cache_set.miss_rate,
                ]
            )


def save_sweep_json(results: list[SweepResult], filename: Path) -> None:
    """Saves cache simulation sweep results to a JSON file.

    Serializes a list of SweepResult objects into a structured JSON format,
    capturing both the cache configuration parameters and their corresponding
    performance metrics (hits, misses, rates, etc.).

    Args:
        results (list[SweepResult]): A list of SweepResult objects containing
            the configuration and performance metrics for each simulation run.
        filename (Path): The Path object representing the target file destination
            where the JSON data will be written.
    """
    output = {
        "sweep_results": [
            {
                "block_size": result.block_size,
                "num_blocks": result.num_blocks,
                "associativity": result.associativity,
                "replacement_policy": result.replacement_policy,
                "total_accesses": result.total_accesses,
                "total_hits": result.total_hits,
                "total_misses": result.total_misses,
                "hit_rate": result.hit_rate,
                "miss_rate": result.miss_rate,
            }
            for result in results
        ]
    }

    with filename.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def save_sweep_csv(results: list[SweepResult], filename: Path) -> None:
    """Saves cache simulation sweep results to a CSV file.

    Serializes a list of SweepResult objects into a tabular CSV format,
    capturing both the cache configuration parameters and their corresponding
    performance metrics as row data.

    Args:
        results: A list of SweepResult objects containing
            the configuration and performance metrics for each simulation run.
        filename: The Path object representing the target file destination
            where the CSV data will be written.
    """
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "block_size",
                "num_blocks",
                "associativity",
                "replacement_policy",
                "total_accesses",
                "total_hits",
                "total_misses",
                "hit_rate",
                "miss_rate",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result.block_size,
                    result.num_blocks,
                    result.associativity,
                    result.replacement_policy,
                    result.total_accesses,
                    result.total_hits,
                    result.total_misses,
                    result.hit_rate,
                    result.miss_rate,
                ]
            )


def generate_plots(results: list[SweepResult], output_dir: Path) -> None:
    """Generate and save performance visualizations from sweep results.

    Creates line charts and heatmaps showing how cache configuration
    parameters affect hit rates.

    Args:
        results: Sweep results collected from multiple cache simulations.
        output_dir: Directory where plot images will be written.

    Raises:
        ValueError: If results is empty.
        OSError: If plots cannot be written.
    """
    if not results:
        raise ValueError("Cannot generate plots from an empty results list.")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(result) for result in results])

    # Plot 1: Associativity vs Hit Rate
    plt.figure(figsize=(8, 5))

    assoc_avg = df.groupby("associativity")["hit_rate"].mean().reset_index()

    plt.plot(
        assoc_avg["associativity"].to_numpy(),
        assoc_avg["hit_rate"].to_numpy(),
        marker="o",
    )

    plt.xlabel("Associativity")
    plt.ylabel("Hit Rate (%)")
    plt.title("Associativity vs Hit Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "associativity_vs_hit_rate.png")
    plt.close()

    # Plot 2: Block Size vs Hit Rate
    plt.figure(figsize=(8, 5))

    block_avg = df.groupby("block_size")["hit_rate"].mean().reset_index()

    plt.plot(
        block_avg["block_size"].to_numpy(), block_avg["hit_rate"].to_numpy(), marker="o"
    )

    plt.xlabel("Block Size (Bytes)")
    plt.ylabel("Hit Rate (%)")
    plt.title("Block Size vs Hit Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "block_size_vs_hit_rate.png")
    plt.close()

    # Plot 3: Number of Blocks vs Hit Rate
    plt.figure(figsize=(8, 5))

    blocks_avg = df.groupby("num_blocks")["hit_rate"].mean().reset_index()

    plt.plot(
        blocks_avg["num_blocks"].to_numpy(),
        blocks_avg["hit_rate"].to_numpy(),
        marker="o",
    )

    plt.xlabel("Number of Blocks")
    plt.ylabel("Hit Rate (%)")
    plt.title("Number of Blocks vs Hit Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "num_blocks_vs_hit_rate.png")
    plt.close()

    # Plot 4: Hit Rate vs Number of Blocks by Associativity
    plt.figure(figsize=(10, 6))

    for assoc in sorted(df["associativity"].unique()):
        subset = df[df["associativity"] == assoc].sort_values("num_blocks")

        plt.plot(
            subset["num_blocks"].to_numpy(),
            subset["hit_rate"].to_numpy(),
            marker="o",
            label=f"{assoc}-way",
        )

    plt.xlabel("Number of Blocks")
    plt.ylabel("Hit Rate (%)")
    plt.title("Hit Rate vs Number of Blocks by Associativity")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "hit_rate_by_associativity.png")
    plt.close()

    # Plot 5: Replacement Policy Comparison
    policy_avg = df.groupby("replacement_policy")["hit_rate"].mean().reset_index()

    plt.figure(figsize=(8, 5))

    plt.bar(
        policy_avg["replacement_policy"].tolist(), policy_avg["hit_rate"].to_numpy()
    )

    plt.xlabel("Replacement Policy")
    plt.ylabel("Average Hit Rate (%)")
    plt.title("Replacement Policy Comparison")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "replacement_policy_comparison.png")
    plt.close()

    # Heatmap 1: Associativity vs Block Size
    pivot = df.pivot_table(
        values="hit_rate", index="associativity", columns="block_size", aggfunc="mean"
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(pivot.to_numpy(), aspect="auto")

    plt.colorbar(label="Hit Rate (%)")

    plt.xlabel("Block Size (Bytes)")
    plt.ylabel("Associativity")
    plt.title("Hit Rate: Associativity vs Block Size")

    plt.xticks(np.arange(len(pivot.columns)), list(pivot.columns))

    plt.yticks(np.arange(len(pivot.index)), list(pivot.index))

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_assoc_vs_block.png")
    plt.close()

    # Heatmap 2: Associativity vs Number of Blocks
    pivot = df.pivot_table(
        values="hit_rate", index="associativity", columns="num_blocks", aggfunc="mean"
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(pivot.to_numpy(), aspect="auto")

    plt.colorbar(label="Hit Rate (%)")

    plt.xlabel("Number of Blocks")
    plt.ylabel("Associativity")
    plt.title("Hit Rate: Associativity vs Number of Blocks")

    plt.xticks(np.arange(len(pivot.columns)), list(pivot.columns))

    plt.yticks(np.arange(len(pivot.index)), list(pivot.index))

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_assoc_vs_blocks.png")
    plt.close()

    # Heatmap 3: Block Size vs Number of Blocks
    pivot = df.pivot_table(
        values="hit_rate", index="block_size", columns="num_blocks", aggfunc="mean"
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(pivot.to_numpy(), aspect="auto")

    plt.colorbar(label="Hit Rate (%)")

    plt.xlabel("Number of Blocks")
    plt.ylabel("Block Size (Bytes)")
    plt.title("Hit Rate: Block Size vs Number of Blocks")

    plt.xticks(np.arange(len(pivot.columns)), list(pivot.columns))

    plt.yticks(np.arange(len(pivot.index)), list(pivot.index))

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_block_vs_blocks.png")
    plt.close()

    # Heatmap 4: Policy vs Associativity
    pivot = df.pivot_table(
        values="hit_rate",
        index="replacement_policy",
        columns="associativity",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(pivot.to_numpy(), aspect="auto")

    plt.colorbar(label="Hit Rate (%)")

    plt.xlabel("Associativity")
    plt.ylabel("Replacement Policy")
    plt.title("Hit Rate: Policy vs Associativity")

    plt.xticks(np.arange(len(pivot.columns)), list(pivot.columns))

    plt.yticks(np.arange(len(pivot.index)), list(pivot.index))

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_policy_vs_associativity.png")
    plt.close()


def main() -> None:
    """Orchestrate the primary execution pipeline of the cache simulator.

    Coordinates command-line argument processing, enforces hardware architectural
    validations, streams and decodes raw memory traces, executes the cache lookup
    and replacement algorithms, and routes final statistics to console outputs and
    if specified by the user, to CSV and/or JSON files.

    Raises:
        ValueError: If any of the cache parameters are invalid (e.g., non-positive block size, associativity not a power of two, etc.)
        FileNotFoundError: If the specified datafile does not exist.
    """
    args = parse_args()
    try:
        config = build_config(args)
        validate_config(config)
    except (ValueError, FileNotFoundError) as e:
        print(f"Input validation error: {e}")
        return

    addresses = load_addresses(config.datafile)

    if args.sweep:
        results = run_sweep(addresses, args)
        print_sweep_results(results)
        best = max(results, key=lambda r: r.hit_rate)

        print("\nBest Configuration")
        print("------------------")
        print(f"Block Size      : {best.block_size}")
        print(f"Num Blocks      : {best.num_blocks}")
        print(f"Associativity   : {best.associativity}")
        print(f"Policy          : {best.replacement_policy}")
        print(f"Hit Rate        : {best.hit_rate:.2f}%")

        if args.json_output:
            save_sweep_json(results, Path(args.json_output))
            print(f"Results saved to {Path(args.json_output)}")

        if args.csv_output:
            save_sweep_csv(results, Path(args.csv_output))
            print(f"Results saved to {Path(args.csv_output)}")

        # Automatically generate and save plots during a sweep
        output_plots_dir = Path(
            "docs/images"
        )  # Nesting inside a clean portfolio directory
        try:
            generate_plots(results, output_plots_dir)
            print(
                f"Sweep visualization plots successfully saved to: '{output_plots_dir}/'"
            )
        except (ValueError, OSError) as e:
            print(f"Warning: Visualizations skipped or failed to write: {e}")

        return

    # Standard Single Simulation Pipeline Execution
    stats = simulate_cache(addresses, config)
    print_statistics(stats, config)

    # Convert raw string paths to Path objects for output functions
    if args.json_output:
        json_path = Path(args.json_output)
        save_json(stats, config, json_path)
        print(f"Results saved to {json_path}")
    if args.csv_output:
        csv_path = Path(args.csv_output)
        save_csv(stats, config, csv_path)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()

# COMPLETED TODO 1: Add sweep mode to run multiple configurations and compare results, creating a performance matrix
# COMPLETED TODO 2: Create visualizations to analyze how different parameters affect hit/miss rates
# COMPLETED TODO 3: Generate plots for GitHub README using matplotlib
# COMPLETED TODO 4: Test with large trace file
# TODO 5: Write README.md for GitHub
