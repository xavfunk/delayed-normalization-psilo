
import math
n_TRs = 268
start_with_phase = 1
block_length_TR = 10
one_block_onoff = block_length_TR*2

print(f"TRs {n_TRs}, block length {block_length_TR} amount of blocks: {268/one_block_onoff}, floored {math.floor(268/one_block_onoff)}")

n_blocks = 8

print(f"start_with_phase == 1: block length {block_length_TR} amount of blocks: {n_blocks}, TRs {one_block_onoff*n_blocks}")
print(f"start_with_phase == 2: block length {block_length_TR} amount of blocks: {n_blocks}, TRs {one_block_onoff*n_blocks-block_length_TR}")
