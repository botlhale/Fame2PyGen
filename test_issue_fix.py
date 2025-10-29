"""
Test script to validate the FAME script interpretation fixes.
"""

from fame2pygen.fame2py_converter import generate_formulas_file, generate_test_script
from fame2pygen.formulas_generator import parse_fame_formula, convert_fame_date_to_iso

def test_date_conversion():
    """Test FAME date conversion to ISO format."""
    print("Testing date conversion...")
    test_cases = [
        ('12jul1985', '1985-07-12'),
        ('13jul1985', '1985-07-13'),
        ('01Jan2020', '2020-01-01'),
        ('2020Q1', '2020-01-01'),
        ('2020-01-01', '2020-01-01'),
    ]
    
    for fame_date, expected_iso in test_cases:
        result = convert_fame_date_to_iso(fame_date)
        assert result == expected_iso, f"Failed: {fame_date} -> {result} (expected {expected_iso})"
        print(f"  ✓ {fame_date} -> {result}")
    print("Date conversion tests passed!\n")


def test_point_in_time_parsing():
    """Test parsing of point-in-time assignments."""
    print("Testing point-in-time assignment parsing...")
    test_cases = [
        ('set a.bot[12jul1985]=130', {
            'type': 'point_in_time_assign',
            'target': 'a.bot',
            'date': '12jul1985',
            'rhs': '130',
        }),
        ('set a.bot[13jul1985]=901', {
            'type': 'point_in_time_assign',
            'target': 'a.bot',
            'date': '13jul1985',
            'rhs': '901',
        }),
    ]
    
    for cmd, expected in test_cases:
        result = parse_fame_formula(cmd)
        for key, value in expected.items():
            assert result.get(key) == value, f"Failed: {cmd} - {key} = {result.get(key)} (expected {value})"
        print(f"  ✓ {cmd}")
    print("Point-in-time parsing tests passed!\n")


def test_case_insensitive_time_index():
    """Test that time indices support both T and t."""
    print("Testing case-insensitive time indices...")
    test_cases = [
        'b_c = d.har[T-1]',
        'b_c = d.har[t-1]',
        'b_c = d.har[T+1]',
        'b_c = d.har[t+1]',
    ]
    
    for cmd in test_cases:
        result = parse_fame_formula(cmd)
        assert result is not None, f"Failed to parse: {cmd}"
        assert result['type'] == 'simple', f"Wrong type for: {cmd}"
        print(f"  ✓ {cmd}")
    print("Case-insensitive time index tests passed!\n")


def test_full_integration():
    """Test the full integration with the problematic FAME script."""
    print("Testing full integration...")
    
    fame_commands = [
        'a.bot=z.some',
        'set a.bot[12jul1985]=130',
        'set a.bot[13jul1985]=901',
        'b_c = (d.har[T-1]/d.har)*(convert(da_val,bus,disc,ave))'
    ]
    
    # Generate files
    generate_formulas_file(fame_commands, 'formulas_test_output.py')
    generate_test_script(fame_commands, 'ts_transformer_test_output.py')
    
    # Read generated transformer
    with open('ts_transformer_test_output.py', 'r') as f:
        content = f.read()
    
    # Check for expected patterns
    checks = [
        ('pl.col("Z.SOME").alias("A.BOT")', 'Initial assignment to A.BOT'),
        ('CONVERT(pl.col("DA_VAL"), "bus", "disc", "ave")', 'Convert function with quoted parameters'),
        ('pl.col("D.HAR").shift(1)', 'Time-indexed reference with capital T'),
        ('pl.when(pl.col("DATE") == pl.lit("1985-07-12").cast(pl.Date))', 'Point-in-time assignment for 12jul1985'),
        ('.when(pl.col("DATE") == pl.lit("1985-07-13").cast(pl.Date))', 'Point-in-time assignment for 13jul1985'),
        ('.then(pl.lit(130))', 'First date assignment value'),
        ('.then(pl.lit(901))', 'Second date assignment value'),
        ('.otherwise(pl.col("A.BOT"))', 'Preserve existing values for other dates'),
    ]
    
    for pattern, description in checks:
        assert pattern in content, f"Missing pattern: {description}\nPattern: {pattern}"
        print(f"  ✓ {description}")
    
    print("\nGenerated code preview:")
    print("=" * 60)
    print(content)
    print("=" * 60)
    print("\nFull integration test passed!\n")


if __name__ == "__main__":
    test_date_conversion()
    test_point_in_time_parsing()
    test_case_insensitive_time_index()
    test_full_integration()
    print("\n✅ All tests passed successfully!")
