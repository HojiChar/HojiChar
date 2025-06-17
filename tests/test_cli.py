import json
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def current_dir():
    return Path(__file__).parent


def test_cli_filter_profile(current_dir):
    test_profile = current_dir / "fixtures/sample_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"
    # test_output_err = current_dir / "fixtures/sample_out_100_stderr.txt"

    result = subprocess.run(
        ["hojichar", "-p", test_profile, "-j", "1"],
        input=open(test_input).read(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == open(test_output).read()


@pytest.mark.parametrize("num_jobs", [1, 2, 4, 8])
def test_cli_filter_profile_multi_jobs(current_dir, num_jobs):
    test_profile = current_dir / "fixtures/sample_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"
    test_output_err = current_dir / "fixtures/sample_out_100_stderr.txt"

    with tempfile.NamedTemporaryFile("w+") as tmpf:
        result = subprocess.run(
            ["hojichar", "-p", test_profile, "-j", str(num_jobs), "--dump-stats", tmpf.name],
            input=open(test_input).read(),
            capture_output=True,
            text=True,
        )
        tmpf.seek(0)
        stats = tmpf.read()
    assert result.returncode == 0
    assert set(result.stdout.split("\n")) == set(open(test_output).read().split("\n"))

    result_stats = json.loads(stats).get("total_info")
    expected_stats = json.loads(open(test_output_err).read()).get("total_info")

    for key, val in result_stats.items():
        if key == "cumulative_time":
            continue

        assert val == expected_stats.get(key)


def test_cli_factory_profile(current_dir):
    test_profile = current_dir / "fixtures/sample_args_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"

    arg = "<hojichar>"
    result = subprocess.run(
        ["hojichar", "-p", test_profile, "--args", arg, "-j", "1"],
        input=open(test_input).read(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == open(test_output).read()


@pytest.mark.parametrize("num_jobs", [1, 2, 4, 8])
def test_cli_factory_profile_multi_jobs(current_dir, num_jobs):
    test_profile = current_dir / "fixtures/sample_args_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"

    arg = "<hojichar>"
    result = subprocess.run(
        ["hojichar", "-p", test_profile, "--args", arg, "-j", str(num_jobs)],
        input=open(test_input).read(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert set(result.stdout.split("\n")) == set(open(test_output).read().split("\n"))


def test_cli_factory_profile_arg2(current_dir):
    test_profile = current_dir / "fixtures/sample_args2_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"

    arg1 = "<hoji"
    arg2 = "char>"
    result = subprocess.run(
        ["hojichar", "-p", test_profile, "-j", "1", "--args", arg1, arg2],
        input=open(test_input).read(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == open(test_output).read()


@pytest.mark.parametrize("num_jobs", [1, 2, 4, 8])
def test_cli_factory_profile_arg2_multi_jobs(current_dir, num_jobs):
    test_profile = current_dir / "fixtures/sample_args2_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"

    arg1 = "<hoji"
    arg2 = "char>"
    result = subprocess.run(
        ["hojichar", "-p", test_profile, "-j", str(num_jobs), "--args", arg1, arg2],
        input=open(test_input).read(),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert set(result.stdout.split("\n")) == set(open(test_output).read().split("\n"))


def test_cli_file_output(current_dir):
    test_profile = current_dir / "fixtures/sample_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"

    with tempfile.NamedTemporaryFile("w+") as tempf:
        hojichar_cmd = ["hojichar", "-p", test_profile, "-o", tempf.name, "-j", "1"]
        process = subprocess.run(hojichar_cmd, input=open(test_input).read(), text=True)

        assert process.returncode == 0
        tempf.seek(0)
        output = tempf.read()
    assert output == open(test_output).read()


@pytest.mark.parametrize("num_jobs", [1, 2, 4, 8])
def test_cli_file_output_multi_jobs(current_dir, num_jobs):
    test_profile = current_dir / "fixtures/sample_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output = current_dir / "fixtures/sample_out_100.jsonl"

    with tempfile.NamedTemporaryFile("w+") as tempf:
        hojichar_cmd = ["hojichar", "-p", test_profile, "-o", tempf.name, "-j", str(num_jobs)]
        process = subprocess.run(hojichar_cmd, input=open(test_input).read(), text=True)

        assert process.returncode == 0
        tempf.seek(0)
        output = tempf.read()
    assert set(output.split("\n")) == set(open(test_output).read().split("\n"))


def test_cli_dump_stats(current_dir):
    test_profile = current_dir / "fixtures/sample_profile.py"
    test_input = current_dir / "fixtures/sample_in_100.jsonl"
    test_output_err = current_dir / "fixtures/sample_out_100_stderr.txt"

    with tempfile.NamedTemporaryFile("w+") as tempf:
        hojichar_cmd = ["hojichar", "-p", test_profile, "--dump-stats", tempf.name]
        process = subprocess.run(hojichar_cmd, input=open(test_input).read(), text=True)

        assert process.returncode == 0
        tempf.seek(0)
        result_stats = json.loads(tempf.read()).get("total_info")
    expected_stats = json.loads(open(test_output_err).read()).get("total_info")

    for key, val in result_stats.items():
        if key == "cumulative_time":
            continue

        assert val == expected_stats.get(key)


def test_cli_error(current_dir):
    test_profile = current_dir / "fixtures/sample_raises_profile.py"
    input = """\
Line1
Line2<raise>
Line3
"""
    expected_output = """\
Line1
Line3
"""

    hojichar_cmd = ["hojichar", "-p", test_profile, "-j", "1"]
    process = subprocess.run(hojichar_cmd, input=input, capture_output=True, text=True)
    assert process.returncode == 0
    assert process.stdout == expected_output


def test_cli_error_exit(current_dir):
    test_profile = current_dir / "fixtures/sample_raises_profile.py"
    input = """\
Line1
Line2<raise>
Line3
"""

    hojichar_cmd = ["hojichar", "-p", test_profile, "--exit-on-error", "-j", "1"]
    process = subprocess.run(hojichar_cmd, input=input, capture_output=True, text=True)
    assert process.returncode == 1
