#!/usr/bin/env python3
"""Notify the JAX chat on MS Teams when upstrem CI breaks"""
import argparse
import logging

import requests

ACTIONS_WORKFLOW_URL = "https://api.github.com/repos/jax-ml/jax/actions/workflows/rocm-ci.yml/runs"
logger = logging.getLogger(__name__)


def get_workflow_status():
    # Get a list of the latest runs from Github. Github will order the results by what time the
    # run was started with the most recent runs at the top.
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    params = {
        "status": "completed",
    }
    resp = requests.get(url=ACTIONS_WORKFLOW_URL, headers=headers, params=params)
    resp.raise_for_status()
    workflow_runs = resp.json().get("workflow_runs", [])
    logger.info("Found %i recent workflow runs", len(workflow_runs))
    logger.debug("Workflow Runs: %s", workflow_runs)

    # Find the most recent run with a status that can be cleanly mapped to a pass or a fail. Runs
    # can have other statuses, like 'cancelled' or 'skipped' that we don't care about.
    for run in workflow_runs:
        status = run.get("conclusion")
        if status == "success":
            return "pass"
        if status == "failure" or status == "timed_out":
            return "fail"
    # If we can't find a workflow with
    raise Exception(
        "Could not find a recent workflow with a 'success', 'failure', or 'timed_out' status"
    )


def get_previous_workflow_status(status_gh_var, gh_token):
    url = f"https://api.github.com/repos/rocm/rocm-jax/actions/variables/{status_gh_var}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {gh_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(url=url, headers=headers)
    resp.raise_for_status()
    print(resp)
    status = resp.json().get("value")
    if status != "pass" or status != "fail":
        logger.warning(
            "Stored previous status was %s. There may be an error when setting %s.",
            status,
            status_gh_var,
        )
        return "pass"
    return status


def save_workflow_status(status, status_gh_var, gh_token):
    assert status == "pass" or status == "fail"
    url = f"https://api.github.com/repos/rocm/rocm-jax/actions/variables/{status_gh_var}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {gh_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {
        "name": status_gh_var,
        "value": status,
    }
    resp = requests.patch(url=url, headers=headers, body=body)
    resp.raise_for_status()


def notify_teams(webhook_url, status):
    if status == "pass":
        text = "🤬 Upstream CI is failing, see [upstream's actions tab](https://github.com/jax-ml/jax/actions/workflows/rocm-ci.yml)"
    elif status == "fail":
        text = "😁 Upstream CI is back to normal, see [upstream's actions tab](https://github.com/jax-ml/jax/actions/workflows/rocm-ci.yml)"
    else:
        raise ValueError("status must be either 'pass' or 'fail'")
    body = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.2",
                    "body": [{"type": "TextBlock", "text": text}],
                },
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url=webhook_url, headers=headers, json=body)
    resp.raise_for_status()


def main(teams_url, status_gh_var, gh_token):
    # Get the current and previous status of the last Actions CI job in upstream
    current_status = get_workflow_status()
    previous_status = get_previous_workflow_status(status_gh_var, gh_token)
    logger.info(
        "Current status is: %s. Previous status was: %s",
        current_status,
        previous_status,
    )

    # If the status has changed, send a notification to MS Teams
    try:
        if current_ci_status != previous_ci_status:
            logger.info("Notifying JAX channel of new stats: %s", current_ci_status)
            notify_teams(teams_url, current_status)
    # Save the current upstream status as the current status
    finally:
        save_workflow_status(previous_ci_status, status_gh_var, gh_token)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teams-url",
        required=True,
        help="URL to the MS Teams webhook that handles notifying the proper chats when CI fails",
    )
    parser.add_argument(
        "--status-gh-var",
        default="UPSTREAM_CI_STATUS",
        help="Github environment variable that stores the CI status between runs of this script",
    )
    parser.add_argument(
        "--gh-token",
        required=True,
        help="Github auth token. Must have permissions to read and write repo variables.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args.teams_url, args.status_gh_var, args.gh_token)
