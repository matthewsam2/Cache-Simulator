import argparse
import os
import random

from dataclasses import dataclass, field
from enum import Enum

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

def validate_inputs(args):

  if args.block_size <= 0:
    raise ValueError("Block size must be a positive integer.")
    
  if args.num_blocks <= 0:
    raise ValueError("Number of blocks must be a positive integer.")
   
  if args.num_blocks % args.associativity != 0:
    raise ValueError("Number of blocks must be a multiple of associativity.")
    
  num_sets = args.num_blocks // args.associativity

  if not is_power_of_two(args.block_size):
    raise ValueError("Block size must be a power of two.")
    
  if not is_power_of_two(num_sets):
    raise ValueError("Number of sets (num_blocks / associativity) must be a power of two.")
    
  if not os.path.isfile(args.datafile):
    raise FileNotFoundError(f"Datafile '{args.datafile}' not found.")
    
def parse_args():
   
  parser = argparse.ArgumentParser(description = "Cache Simulator")
  parser.add_argument("--block_size", type = int, default = 64, help = "Block size in bytes (must be > 0 and a power of two)")
  parser.add_argument("--num_blocks", type = int, default = 256, help = "Number of blocks in the cache (must be > 0)")
  parser.add_argument( # Broken into multiple lines for long help message
    "--associativity",
    type = int,
    choices = [1, 2, 4, 8, 16],
    default = 1,
    help = "Cache associativity (1 = direct-mapped, 2 = 2-way, 4 = 4-way, 8 = 8-way, 16 = 16-way)"
  )
  parser.add_argument("--replacement_policy", type = ReplacementPolicy, choices = [p.value for p in ReplacementPolicy], default = ReplacementPolicy.LRU.value, help = "Replacement policy (LRU, FIFO, or random)")
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

def load_addresses(filename: str):

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

def decode_address(address: int, config: CacheConfig):

  offset_bits = config.offset_bits
  index_bits = config.index_bits
  set_index = (address >> offset_bits) & ((1 << index_bits) - 1) # Extract set index
  tag = address >> (offset_bits + index_bits) # Extract tag

  return tag, set_index

def simulate_cache(addresses: list[int], config: CacheConfig):

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

def main():
  args = parse_args()
  try:
    validate_inputs(args)
  except (ValueError, FileNotFoundError) as e:
    print(f"Input validation error: {e}")
    return

  config = build_config(args)
  addresses = load_addresses(args.datafile)
  stats = simulate_cache(addresses, config)

  print(f'Accesses: {stats.total_accesses}, Hits: {stats.total_hits}, Misses: {stats.total_misses}, Hit Rate: {stats.hit_rate:.2f}%, Miss Rate: {stats.miss_rate:.2f}%')

if __name__ == "__main__":
  main()

# TODO 1: Gather statistics and output results in JSON/CSV formats if specified by user
# TODO 2: Add sweep mode to run multiple configurations and compare results
# TODO 3: Create performance matrix and visualizations to analyze how different parameters affect hit/miss rates
# TODO 4: Generate plots for GitHub README using matplotlib
# TODO 5: Celebrate turning an old lab assignment into something good for a portfolio