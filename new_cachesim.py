import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ReplacementPolicy(Enum): # Enum for replacement policies
  """Enumeration of supported cache block eviction strategies.

  Attributes:
    LRU: Least Recently Used - evicts the block that has not been accessed for the longest time.
    FIFO: First-In-First-Out - evicts the block that was inserted earliest into the set.
    RANDOM: Random - evicts a randomly selected block from the set.
  """
  LRU = "LRU"
  FIFO = "FIFO"
  RANDOM = "random"

@dataclass
class CacheConfig: # Holds simulator settings in one place
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
    return self.block_size.bit_length() - 1 # Number of bits for block offset
  
  @property
  def index_bits(self) -> int:
    return self.num_sets.bit_length() - 1 # Number of bits for set index

@dataclass
class CacheBlock: # Represents a single cache line/block
  """Represents a single cache line or block within a set.

  Attributes:
    tag: The unique memory address identifier stored in this block.
    last_used: A cycle timestamp tracking the most recent access (for LRU).
    insertion_time: A cycle timestamp tracking when the block arrived (for FIFO).
  """
  tag: int
  last_used: int = 0 # Used by LRU
  insertion_time: int = 0 # Used by FIFO

@dataclass
class CacheSet: # Represents a set in the cache
  """Represents a set containing multiple cache blocks.

  Attributes:
    blocks: A list of cache blocks in this set.
    hits: The number of cache hits in this set.
    misses: The number of cache misses in this set.
  """
  blocks: list[CacheBlock] = field(default_factory = list)
  hits: int = 0
  misses: int = 0

  @property
  def total_accesses(self) -> int:
    return self.hits + self.misses
  
  @property
  def hit_rate(self) -> float:
    return (self.hits / self.total_accesses) * 100 if self.total_accesses > 0 else 0.0
  
  @property
  def miss_rate(self) -> float:
    return (self.misses / self.total_accesses) * 100 if self.total_accesses > 0 else 0.0

@dataclass
class CacheStats: # Holds overall cache performance statistics
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
    return (self.total_hits / self.total_accesses) * 100 if self.total_accesses > 0 else 0.0
  
  @property
  def miss_rate(self) -> float:
    return (self.total_misses / self.total_accesses) * 100 if self.total_accesses > 0 else 0.0

def is_power_of_two(value: int) -> bool:
  """Checks if a number is an integral power of two using bitwise operations.

  Args:
    value: The integer number to check

  Returns:
    True if the number is a power of two, False otherwise
  """
  # A power of two has exactly one bit set to '1' in its binary representation.
  # Subtracting 1 flips all bits up to that '1', causing bitwise AND to yield 0 for powers of two.
  return value > 0 and (value & (value - 1)) == 0

def validate_config(config: CacheConfig) -> None:
  """Validates the cache architecture configuration parameters and input file.

  Ensures that all parameters adhere to hardware simulation limits, geometric
  constraints, and that the trace file exists.

  Args:
    config (CacheConfig): A CacheConfig object containing the cache parameters and datafile path.

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
    raise ValueError("Number of sets (num_blocks / associativity) must be a power of two.")
    
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
   
  parser = argparse.ArgumentParser(description = "Cache Simulator")
  parser.add_argument("--block_size", type = int, default = 64, help = "Block size in bytes (must be > 0 and a power of two)")
  parser.add_argument("--num_blocks", type = int, default = 256, help = "Number of blocks in the cache (must be > 0)")
  parser.add_argument( # Broken into multiple lines for long help message
    "--associativity",
    type = int,
    default = 1,
    help = "Cache associativity (1 = direct-mapped, 2 = 2-way, 4 = 4-way, etc. Must be a power of two and less than or equal to num_blocks.)"
  )
  parser.add_argument("--replacement_policy", choices = [p.value for p in ReplacementPolicy], default = ReplacementPolicy.LRU.value, help = "Replacement policy (LRU, FIFO, or random)")
  parser.add_argument("--datafile", type = str, required = True, help = "File with memory addresses (one address per line)")
  parser.add_argument("--json_output", type = str, help = "Optional JSON file to save results")
  parser.add_argument("--csv_output", type = str, help = "Optional CSV file to save results")

  return parser.parse_args()

def build_config(args: argparse.Namespace) -> CacheConfig:
  """Construct a validated CacheConfig object from parsed command-line arguments.

  Extracts raw specifications and path boundaries from the CLI namespace and
  initializes them into structured simulator parameters.

  Args:
    args: The parsed command-line argument namespace containing attributes for
          block_size, num_blocks, associativity, replacement_policy, and datafile.

  Returns:
    A structured dataclass instance containing the parsed configuration ready for validation.
  """
  return CacheConfig(
    block_size = args.block_size,
    num_blocks = args.num_blocks,
    associativity = args.associativity,
    replacement_policy = ReplacementPolicy(args.replacement_policy),
    datafile = Path(args.datafile)
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
    for line_number, line in enumerate(f, start = 1):
      line = line.strip()
      if not line:
        continue # Skip empty lines
      try:
        addresses.append(int(line, 16)) # Convert hex string to integer
      except ValueError:
        raise ValueError(f"Invalid address '{line}' on line {line_number} in datafile. Must be a valid hexadecimal number.")
      
  return addresses

def decode_address(address: int, config: CacheConfig) -> tuple[int, int]:
  """Extract the tag and set index fields from a raw memory address.

  Uses bitwise right-shifts and generated bitmasks based on the cache's
  configured offset and index bit widths to isolate specific fields.

  Args:
    address: The raw numerical memory address integer to decode.
    config: The structured CacheConfig object providing geometric bit widths.

  Returns:
    A tuple containing:
      tag: The unique identifier tag for the memory block.
      set_index: The target cache set array index where the block resides.
  """

  offset_bits = config.offset_bits
  index_bits = config.index_bits
  set_index = (address >> offset_bits) & ((1 << index_bits) - 1) # Extract set index
  tag = address >> (offset_bits + index_bits) # Extract tag

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
    An aggregated performance statistics collection holding hit counts, miss counts,
    and final structural cache states.
  """

  num_sets = config.num_sets
  cache = [CacheSet() for _ in range(num_sets)]
  total_hits, total_misses = 0, 0
  time = 0 # Global time counter for LRU/FIFO
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
    if hit_block is not None: # Explicit check for None because 0 is a valid block despite being falsy
      total_hits += 1
      cache_set.hits += 1
      if config.replacement_policy == ReplacementPolicy.LRU:
        hit_block.last_used = time # Update usage time for LRU

    # Cache miss
    else:
      total_misses += 1
      cache_set.misses += 1
      new_block = CacheBlock(tag = tag, last_used = time, insertion_time = time)

      # Space available in set
      if len(cache_set.blocks) < config.associativity:
        cache_set.blocks.append(new_block)

      # Need to evict a block
      else:
        if config.replacement_policy == ReplacementPolicy.LRU:
          victim_index = min(range(len(cache_set.blocks)), key = lambda idx: cache_set.blocks[idx].last_used) # Find least recently used block

        elif config.replacement_policy == ReplacementPolicy.FIFO:
          victim_index = min(range(len(cache_set.blocks)), key = lambda idx: cache_set.blocks[idx].insertion_time) # Find first inserted block

        elif config.replacement_policy == ReplacementPolicy.RANDOM:
          victim_index = random.randrange(len(cache_set.blocks)) # Randomly select a block to evict
        
        cache_set.blocks[victim_index] = new_block # Replace victim block with new block
      
  return CacheStats(total_hits = total_hits, total_misses = total_misses, cache = cache)

def print_statistics(stats: CacheStats) -> None:
  """Print the final aggregated simulation performance metrics to the console.

  Outputs structured rows showing total memory access attempts, absolute hit and
  miss counts, and calculated percentage efficiency rates rounded to two decimal places.

  Args:
    stats: The collection containing overall simulator performance metrics.
  """

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
      "replacement_policy": config.replacement_policy.value
    },

    "statistics": {
      "total_accesses": stats.total_accesses,
      "total_hits": stats.total_hits,
      "total_misses": stats.total_misses,
      "hit_rate": stats.hit_rate,
      "miss_rate": stats.miss_rate
    },

    "sets": [
      {
        "set_index": i,
        "hits": cache_set.hits,
        "misses": cache_set.misses,
        "hit_rate": cache_set.hit_rate,
        "miss_rate": cache_set.miss_rate
      }
      for i, cache_set in enumerate(stats.cache)
    ]
  }

  with filename.open("w", encoding = "utf-8") as f:
    json.dump(output, f, indent = 2)

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
    writer.writerow([
      "set_index",
      "hits",
      "misses",
      "hit_rate",
      "miss_rate"
    ])

    for i, cache_set in enumerate(stats.cache):
      writer.writerow([
        i,
        cache_set.hits,
        cache_set.misses,
        cache_set.hit_rate,
        cache_set.miss_rate
      ])

def main() -> None:
  """Orchestrate the primary execution pipeline of the cache simulator.

  Coordinates command-line argument processing, enforces hardware architectural 
  validations, streams and decodes raw memory traces, executes the cache lookup
  and replacement algorithms, and routes final statistics to outputs.
  """
  args = parse_args()
  try:
    config = build_config(args)
    validate_config(config)
  except (ValueError, FileNotFoundError) as e:
    print(f"Input validation error: {e}")
    return

  addresses = load_addresses(config.datafile)
  stats = simulate_cache(addresses, config)
  print_statistics(stats)
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

# TODO 1: Add sweep mode to run multiple configurations and compare results
# TODO 2: Create performance matrix and visualizations to analyze how different parameters affect hit/miss rates
# TODO 3: Generate plots for GitHub README using matplotlib
# TODO 4: Celebrate turning an old lab assignment into something good for a portfolio