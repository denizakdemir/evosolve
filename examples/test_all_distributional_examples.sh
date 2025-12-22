#!/bin/bash
# Test all distributional examples to ensure they run without errors

echo "Testing all distributional examples..."
echo "======================================"
echo

EXAMPLES_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EXAMPLES_DIR/.."

FAILED=0
PASSED=0

test_example() {
    local example=$1
    local name=$(basename "$example")

    echo "Testing: $name"
    if python "$example" > /dev/null 2>&1; then
        echo "  ✓ PASSED"
        ((PASSED++))
    else
        echo "  ✗ FAILED"
        ((FAILED++))
        # Show error for debugging
        echo "  Error output:"
        python "$example" 2>&1 | tail -10 | sed 's/^/    /'
    fi
    echo
}

# Test all distributional examples
test_example "examples/distributional_core_demo.py"
test_example "examples/distributional_binary_optimization.py"
test_example "examples/distributional_presets_demo.py"
test_example "examples/distributional_vs_standard_pareto.py"

echo "======================================"
echo "Results: $PASSED passed, $FAILED failed"
echo

if [ $FAILED -eq 0 ]; then
    echo "✓ All distributional examples work correctly!"
    exit 0
else
    echo "✗ Some examples failed. Please review errors above."
    exit 1
fi
