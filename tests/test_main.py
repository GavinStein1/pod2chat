"""Tests for main module."""

import pytest
from src import main


def test_main_function(capsys):
    """Test that main function prints expected output."""
    main.main()
    captured = capsys.readouterr()
    assert "Hello, World!" in captured.out
    assert "Welcome to Pod2Chat!" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
