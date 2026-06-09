import argparse
import random
import json
import csv

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class ReplacementPolicy(Enum): # Enum for replacement policies
  LRU = "LRU"
  FIFO = "FIFO"
  RANDOM = "random"

@dataclass
class CacheConfig: # Holds simulator settings in one place
  block_size: int
  num_blocks: int
  associativity: int
  replacement_policy: ReplacementPolicy
  datafile: str

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
  tag: int
  last_used: int = 0 # Used by LRU
  insertion_time: int = 0 # Used by FIFO

@dataclass
class CacheSet: # Represents a set in the cache
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
  """Check if an integer is an integral power of two."""
  # A power of two has exactly one bit set to '1' in its binary representation.
  # Subtracting 1 flips all bits up to that '1', causing bitwise AND to yield 0 for powers of two.
  return value > 0 and (value & (value - 1)) == 0

def validate_config(config: CacheConfig):

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

def build_config(args) -> CacheConfig:
  return CacheConfig(
    block_size = args.block_size,
    num_blocks = args.num_blocks,
    associativity = args.associativity,
    replacement_policy = ReplacementPolicy(args.replacement_policy),
    datafile = args.datafile
  )

def load_addresses(filename: str) -> list[int]:

  addresses = []

  with open(filename, "r") as f:
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

  offset_bits = config.offset_bits
  index_bits = config.index_bits
  set_index = (address >> offset_bits) & ((1 << index_bits) - 1) # Extract set index
  tag = address >> (offset_bits + index_bits) # Extract tag

  return tag, set_index

def simulate_cache(addresses: list[int], config: CacheConfig) -> CacheStats:

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

def print_statistics(stats: CacheStats):

  print("\nCache Statistics")
  print("----------------")
  print(f"Accesses : {stats.total_accesses}")
  print(f"Hits     : {stats.total_hits}")
  print(f"Misses   : {stats.total_misses}")
  print(f"Hit Rate : {stats.hit_rate:.2f}%")
  print(f"Miss Rate: {stats.miss_rate:.2f}%")  

def save_json(stats: CacheStats, config: CacheConfig, filename: str) -> None: # Return type hint for future-me

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

  with open(filename, "w") as f:
    json.dump(output, f, indent = 2)

def save_csv(stats: CacheStats, filename: str) -> None:

  with open(filename, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["Set Index", "Hits", "Misses", "Total Accesses", "Hit Rate (%)", "Miss Rate (%)"])

    for idx, cache_set in enumerate(stats.cache):
      writer.writerow([
        idx,
        cache_set.hits,
        cache_set.misses,
        cache_set.total_accesses,
        f"{cache_set.hit_rate:.2f}",
        f"{cache_set.miss_rate:.2f}"
      ])

def main():
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
  if args.json_output:
    save_json(stats, config, args.json_output)
    print(f"Results saved to {args.json_output}")
  if args.csv_output:
    save_csv(stats, args.csv_output)
    print(f"Results saved to {args.csv_output}")

if __name__ == "__main__":
  main()

# COMPLETED TODO 1: Gather statistics and output results in JSON/CSV formats if specified by user
# TODO 2: Add sweep mode to run multiple configurations and compare results
# TODO 3: Create performance matrix and visualizations to analyze how different parameters affect hit/miss rates
# TODO 4: Generate plots for GitHub README using matplotlib
# TODO 5: Celebrate turning an old lab assignment into something good for a portfolio