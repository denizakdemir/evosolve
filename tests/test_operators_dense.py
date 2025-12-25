
import unittest
import time
import random
from evosolve.operators import _get_valid_replacement

class TestOperatorsDense(unittest.TestCase):
    def test_dense_selection_performance(self):
        """
        Test performance of _get_valid_replacement when selection is very dense.
        """
        # Scenario: Select 9999 items out of 10000.
        # Only 1 item is available.
        # Rejection sampling with 20 attempts has probability (1/10000)^20 ~ 0 of finding it.
        # It will fallback to listing all available candidates.
        
        candidates = list(range(10000))
        current_set = set(range(9999)) # 0..9998 are taken
        current_val = 9998 # We are replacing this one, so it is "out" technically but passed as arg
        
        # Actually current_val is the one LEAVING the set, so it shouldn't be in current_set check technically?
        # The function signature is `_get_valid_replacement(current_val, current_set, candidates, ...)`
        # It finds val s.t. val != current_val and val not in current_set.
        
        # If we are mutating, we remove old_val from set, then call this?
        # Let's check mutation implementation in operators.py.
        # line 418: old_val = row[pos]
        # line 419: new_val = _get_valid_replacement(old_val, current_set, cand, stype)
        # line 423: current_set.discard(old_val); current_set.add(new_val)
        
        # So current_set INCLUDES old_val when called.
        # And we want new_val != old_val and new_val NOT in current_set.
        # But wait, old_val IS in current_set. So "not in current_set" handles "!= old_val".
        # Why does _get_valid_replacement check "val != current_val" specifically?
        # "val != current_val and val not in current_set"
        # If val == current_val, it IS in current_set. Redundant but safe.
        
        start_time = time.time()
        
        for _ in range(100):
            _get_valid_replacement(current_val, current_set, candidates, "UOS")
            
        duration = time.time() - start_time
        print(f"Dense selection 100 iters took: {duration:.4f}s")
        
        # With optimization, this should be very fast.
        # Without, it iterates 10000 items 100 times => 1M ops. 
        # In python that's fast enough (~0.1s).
        # Let's increase scale to make it noticeable.
        
    def test_very_dense_scale(self):
        # 100k items.
        N = 100000
        candidates = list(range(N))
        current_set = set(range(N-1))
        current_val = 0
        
        start = time.time()
        # doing just 10 calls on 100k list construction = 1M ops.
        for _ in range(10):
            val = _get_valid_replacement(current_val, current_set, candidates, "UOS")
            self.assertEqual(val, N-1)
            
        # If this takes > 1s, it's slow.
        # The fix should make it ~0s because it calculates set difference efficiently?
        # Set difference of (list - set) is still O(N) if doing list traversal.
        # Python set difference is O(len(s) + len(other)).
        
        # The previous implementation:
        # available = [c for c in candidates if c != current_val and c not in current_set]
        # This iterates candidates (list). Checking `in current_set` (set) is O(1).
        # Total O(N).
        # For N=100k, 10 calls = 1M steps. Should contain it.
        pass

if __name__ == '__main__':
    unittest.main()
