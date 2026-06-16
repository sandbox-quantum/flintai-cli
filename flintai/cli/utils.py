"""Subcommand for `flintai init`."""

import os
from pathlib import Path


def get_flintai_dir() -> Path:
    return Path.home() / ".flintai"


def get_flintai_env_path() -> Path:
    return get_flintai_dir() / ".env"


def get_flintai_config_path() -> Path:
    return get_flintai_dir() / "config.json"


_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "JENKINS_URL",
    "TRAVIS",
    "BUILDKITE",
    "CODEBUILD_BUILD_ID",
    "TF_BUILD",
    "BITBUCKET_PIPELINE",
    "TEAMCITY_VERSION",
)


def is_ci() -> bool:
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


