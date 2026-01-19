#!/usr/bin/env python3
"""Tests for train_conditional_overnight.py"""

import sys
from unittest.mock import patch, MagicMock

import pytest


class TestOvernightTraining:
    """Tests for overnight training script."""

    def test_main_builds_correct_command(self, capsys):
        """Test that main() builds the correct training command."""
        from src.training.train_conditional_overnight import main

        with patch('src.training.train_conditional_overnight.subprocess.run') as mock_run:
            main()

            # Verify subprocess.run was called
            mock_run.assert_called_once()

            # Get the command that was passed
            cmd = mock_run.call_args[0][0]

            # Verify key arguments
            assert "train_conditional.py" in cmd[1]
            assert "--denoiser" in cmd
            assert "checkpoints_long/final.pt" in cmd
            assert "--max_steps" in cmd
            assert "25000" in cmd
            assert "--batch_size" in cmd
            assert "16" in cmd
            assert "--no_amp" in cmd

    def test_main_prints_info(self, capsys):
        """Test that main() prints training info."""
        from src.training.train_conditional_overnight import main

        with patch('src.training.train_conditional_overnight.subprocess.run'):
            main()

        captured = capsys.readouterr()
        assert "Overnight Conditional Training Run" in captured.out
        assert "25,000" in captured.out
        assert "7-8 hours" in captured.out
        assert "checkpoints_conditional_overnight" in captured.out

    def test_command_includes_all_required_args(self):
        """Test that command includes all necessary training arguments."""
        from src.training.train_conditional_overnight import main

        with patch('src.training.train_conditional_overnight.subprocess.run') as mock_run:
            main()

            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)

            # Check all required arguments are present
            required_args = [
                "--denoiser",
                "--data_dir",
                "--max_steps",
                "--batch_size",
                "--learning_rate",
                "--warmup_steps",
                "--eval_every",
                "--save_every",
                "--log_every",
                "--checkpoint_dir",
                "--no_amp",
            ]

            for arg in required_args:
                assert arg in cmd_str, f"Missing argument: {arg}"
