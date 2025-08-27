# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class GithubCreateIssueModel(BaseModel):
    title: str = Field(description="The title of the GitHub Issue")
    body: str = Field(description="The body of the GitHub Issue")


class GithubCreateIssueModelList(BaseModel):
    issues: list[GithubCreateIssueModel] = Field(default_factory=list,
                                                 description=("A list of GitHub issues, "
                                                              "each with a title and a body"))


class GithubGetIssueModel(BaseModel):
    state: Literal["open", "closed", "all"] | None = Field(default="open",
                                                           description="Issue state used in issue query filter")
    assignee: str | None = Field(default=None, description="Assignee name used in issue query filter")
    creator: str | None = Field(default=None, description="Creator name used in issue query filter")
    mentioned: str | None = Field(default=None, description="Name of person mentioned in issue")
    labels: list[str] | None = Field(default=None, description="A list of labels that are assigned to the issue")
    since: str | None = Field(default=None,
                              description="Only show results that were last updated after the given time.")

    @classmethod
    @field_validator('since', mode='before')
    def validate_since(cls, v):
        if v is None:
            return v
        try:
            # Parse the string to a datetime object
            parsed_date = datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
            # Return the formatted string
            return parsed_date.isoformat() + 'Z'
        except ValueError as e:
            raise ValueError("since must be in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ") from e


class GithubGetIssueModelList(BaseModel):
    filter_parameters: list[GithubGetIssueModel] = Field(default_factory=list,
                                                         description=("A list of query params when fetching issues "
                                                                      "each of type GithubGetIssueModel"))


class GithubUpdateIssueModel(BaseModel):
    issue_number: str = Field(description="The issue number that will be updated")
    title: str | None = Field(default=None, description="The title of the GitHub Issue")
    body: str | None = Field(default=None, description="The body of the GitHub Issue")
    state: Literal["open", "closed"] | None = Field(default=None, description="The new state of the issue")

    state_reason: Literal["completed", "not_planned", "reopened"] | None = Field(
        default=None, description="The reason for changing the state of the issue")

    labels: list[str] | None = Field(default=None, description="A list of labels to assign to the issue")
    assignees: list[str] | None = Field(default=None, description="A list of assignees to assign to the issue")


class GithubUpdateIssueModelList(BaseModel):
    issues: list[GithubUpdateIssueModel] = Field(default_factory=list,
                                                 description=("A list of GitHub issues each "
                                                              "of type GithubUpdateIssueModel"))


class GithubCreatePullModel(BaseModel):
    title: str = Field(description="Title of the pull request")
    body: str = Field(description="Description of the pull request")
    source_branch: str = Field(description="The name of the branch containing your changes")
    target_branch: str = Field(description="The name of the branch you want to merge into")
    assignees: list[str] | None = Field(default=None,
                                        description="List of GitHub usernames to assign to the PR. "
                                        "Always the current user")
    reviewers: list[str] | None = Field(default=None, description="List of GitHub usernames to request review from")


class GithubCreatePullList(BaseModel):
    pull_details: list[GithubCreatePullModel] = Field(
        default_factory=list, description=("A list of params used for creating the PR in GitHub"))


class GithubGetPullsModel(BaseModel):
    state: Literal["open", "closed", "all"] | None = Field(default="open",
                                                           description="Issue state used in issue query filter")
    head: str | None = Field(default=None,
                             description="Filters pulls by head user or head organization and branch name")
    base: str | None = Field(default=None, description="Filters pull by branch name")


class GithubGetPullsModelList(BaseModel):
    filter_parameters: list[GithubGetPullsModel] = Field(
        default_factory=list,
        description=("A list of query params when fetching pull requests "
                     "each of type GithubGetPullsModel"))


class GithubCommitCodeModel(BaseModel):
    branch: str = Field(description="The branch of the remote repo to which the code will be committed")
    commit_msg: str = Field(description="Message with which the code will be committed to the remote repo")
    local_path: str = Field(description="Local filepath of the file that has been updated and "
                            "needs to be committed to the remote repo")
    remote_path: str = Field(description="Remote filepath of the updated file in GitHub. Path is relative to "
                             "root of current repository")


class GithubCommitCodeModelList(BaseModel):
    updated_files: list[GithubCommitCodeModel] = Field(default_factory=list,
                                                       description=("A list of local filepaths and commit messages"))


class GithubGroupConfig(FunctionGroupBaseConfig, name="github"):
    """Function group for GitHub repository operations.

    Exposes issue, pull request, and commit operations with shared configuration.
    """
    repo_name: str = Field(description="The repository name in the format 'owner/repo'")
    timeout: int = Field(default=300, description="Timeout in seconds for GitHub API requests")
    # Required for commit function
    local_repo_dir: str | None = Field(default=None,
                                       description="Absolute path to the local clone. Required for 'commit' function")


@register_function_group(config_type=GithubGroupConfig)
async def github_tool(config: GithubGroupConfig, _builder: Builder):
    """Register the `github` function group with shared configuration.

    Implements:
      - create_issue, get_issue, update_issue
      - create_pull, get_pull
      - commit
    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    async with httpx.AsyncClient(timeout=config.timeout,
                                 headers={
                                     "Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"
                                 }) as client:

        # Issues
        async def create_issue(issues_list: GithubCreateIssueModelList) -> str:
            url = f"https://api.github.com/repos/{config.repo_name}/issues"
            results = []
            for issue in issues_list.issues:
                payload = issue.model_dump(exclude_unset=True)
                response = await client.post(url, json=payload)
                response.raise_for_status()
                results.append(response.json())
            return json.dumps(results)

        async def get_issue(issues_list: GithubGetIssueModelList) -> str:
            url = f"https://api.github.com/repos/{config.repo_name}/issues"
            results = []
            for issue in issues_list.filter_parameters:
                params = issue.model_dump(exclude_unset=True, exclude_none=True)
                response = await client.get(url, params=params)
                response.raise_for_status()
                results.append(response.json())
            return json.dumps(results)

        async def update_issue(issues_list: GithubUpdateIssueModelList) -> str:
            base_url = f"https://api.github.com/repos/{config.repo_name}/issues"
            results = []
            for issue in issues_list.issues:
                payload = issue.model_dump(exclude_unset=True)
                issue_number = payload.pop("issue_number")
                issue_url = f"{base_url}/{issue_number}"
                response = await client.patch(issue_url, json=payload)
                response.raise_for_status()
                results.append(response.json())
            return json.dumps(results)

        # Pull requests
        async def create_pull(pull_list: GithubCreatePullList) -> str:
            results = []
            for pull_detail in pull_list.pull_details:
                pr_url = f"https://api.github.com/repos/{config.repo_name}/pulls"
                pr_data = {
                    "title": pull_detail.title,
                    "body": pull_detail.body,
                    "head": pull_detail.source_branch,
                    "base": pull_detail.target_branch,
                }
                pr_response = await client.post(pr_url, json=pr_data)
                pr_response.raise_for_status()
                pr_number = pr_response.json()["number"]

                assignees_response_json = None
                if pull_detail.assignees:
                    assignees_url = f"https://api.github.com/repos/{config.repo_name}/issues/{pr_number}/assignees"
                    assignees_data = {"assignees": pull_detail.assignees}
                    assignees_response = await client.post(assignees_url, json=assignees_data)
                    assignees_response.raise_for_status()
                    assignees_response_json = assignees_response.json()

                reviewers_response_json = None
                if pull_detail.reviewers:
                    reviewers_url = f"https://api.github.com/repos/{config.repo_name}/pulls/{pr_number}/requested_reviewers"
                    reviewers_data = {"reviewers": pull_detail.reviewers}
                    reviewers_response = await client.post(reviewers_url, json=reviewers_data)
                    reviewers_response.raise_for_status()
                    reviewers_response_json = reviewers_response.json()

                result = {
                    "pull_request": pr_response.json(),
                    "assignees": assignees_response_json,
                    "reviewers": reviewers_response_json,
                }
                results.append(result)

            return json.dumps(results)

        async def get_pull(pull_list: GithubGetPullsModelList) -> str:
            url = f"https://api.github.com/repos/{config.repo_name}/pulls"
            results = []
            for pull_params in pull_list.filter_parameters:
                params = pull_params.model_dump(exclude_unset=True, exclude_none=True)
                response = await client.get(url, params=params)
                response.raise_for_status()
                results.append(response.json())

            return json.dumps(results)

        # Commits (commit updated files)
        async def commit(updated_file_list: GithubCommitCodeModelList) -> str:
            if not config.local_repo_dir:
                raise ValueError("'local_repo_dir' must be set in the github function group config to use 'commit'")

            results = []
            for updated_file in updated_file_list.updated_files:
                branch = updated_file.branch
                commit_msg = updated_file.commit_msg
                local_path = updated_file.local_path
                remote_path = updated_file.remote_path

                # Read content from the local file
                local_path = os.path.join(config.local_repo_dir, local_path)
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 1) Create blob
                blob_url = f"https://api.github.com/repos/{config.repo_name}/git/blobs"
                blob_data = {"content": content, "encoding": "utf-8"}
                blob_response = await client.post(blob_url, json=blob_data)
                blob_response.raise_for_status()
                blob_sha = blob_response.json()["sha"]

                # 2) Get base ref
                ref_url = f"https://api.github.com/repos/{config.repo_name}/git/refs/heads/{branch}"
                ref_response = await client.get(ref_url)
                ref_response.raise_for_status()
                base_tree_sha = ref_response.json()["object"]["sha"]

                # 3) Create tree
                tree_url = f"https://api.github.com/repos/{config.repo_name}/git/trees"
                tree_data = {
                    "base_tree": base_tree_sha,
                    "tree": [{
                        "path": remote_path, "mode": "100644", "type": "blob", "sha": blob_sha
                    }],
                }
                tree_response = await client.post(tree_url, json=tree_data)
                tree_response.raise_for_status()
                tree_sha = tree_response.json()["sha"]

                # 4) Create commit
                commit_url = f"https://api.github.com/repos/{config.repo_name}/git/commits"
                commit_data = {"message": commit_msg, "tree": tree_sha, "parents": [base_tree_sha]}
                commit_response = await client.post(commit_url, json=commit_data)
                commit_response.raise_for_status()
                commit_sha = commit_response.json()["sha"]

                # 5) Update ref
                update_ref_url = f"https://api.github.com/repos/{config.repo_name}/git/refs/heads/{branch}"
                update_ref_data = {"sha": commit_sha}
                update_ref_response = await client.patch(update_ref_url, json=update_ref_data)
                update_ref_response.raise_for_status()

                results.append({
                    "blob_resp": blob_response.json(),
                    "original_tree_ref": tree_response.json(),
                    "commit_resp": commit_response.json(),
                    "updated_tree_ref_resp": update_ref_response.json(),
                })

            return json.dumps(results)

    group = FunctionGroup(config=config)

    group.add_function("create_issue",
                       create_issue,
                       description=f"Creates a GitHub issue in the repo named {config.repo_name}",
                       input_schema=GithubCreateIssueModelList)
    group.add_function("get_issue",
                       get_issue,
                       description=f"Fetches a particular GitHub issue in the repo named {config.repo_name}",
                       input_schema=GithubGetIssueModelList)
    group.add_function("update_issue",
                       update_issue,
                       description=f"Updates a GitHub issue in the repo named {config.repo_name}",
                       input_schema=GithubUpdateIssueModelList)
    group.add_function("create_pull",
                       create_pull,
                       description="Creates a pull request with assignees and reviewers in"
                       f"the GitHub repository named {config.repo_name}",
                       input_schema=GithubCreatePullList)
    group.add_function("get_pull",
                       get_pull,
                       description="Fetches the files for a particular GitHub pull request"
                       f"in the repo named {config.repo_name}",
                       input_schema=GithubGetPullsModelList)
    group.add_function("commit",
                       commit,
                       description="Commits and pushes modified code to a GitHub repository"
                       f"in the repo named {config.repo_name}",
                       input_schema=GithubCommitCodeModelList)

    yield group


class GithubFilesGroupConfig(FunctionGroupBaseConfig, name="github_files"):
    timeout: int = Field(default=5, description="Timeout in seconds for HTTP requests")


@register_function_group(config_type=GithubFilesGroupConfig)
async def github_files_tool(config: GithubFilesGroupConfig, _builder: Builder):

    import re

    import httpx

    async with httpx.AsyncClient(timeout=config.timeout) as client:

        async def get(url_text: str) -> str:

            pattern = r"https://github.com/(.*)/blob/(.*)"
            matches = re.findall(pattern, url_text)
            if len(matches) == 0:
                return ("Invalid github url. Please provide a valid github url. "
                        "Example: 'https://github.com/my_repository/blob/main/file.txt'")

            raw_url = f"https://raw.githubusercontent.com/{matches[0][0]}/refs/heads/{matches[0][1]}"
            try:
                response = await client.get(raw_url)
                response.raise_for_status()
            except httpx.TimeoutException:
                return f"Timeout encountered when retrieving resource: {raw_url}"

            return f"```{response.text}\n```"

        async def get_line(url_text: str) -> str:
            pattern = r"https://github.com/(.*)/blob/(.*)(#L(\d+))"
            matches = re.findall(pattern, url_text)

            if len(matches) == 0:
                return ("Invalid github url. Please provide a valid github url with line information. "
                        "Example: 'https://github.com/my_repository/blob/main/file.txt#L409'")

            line_number = int(matches[0][3])
            raw_url = f"https://raw.githubusercontent.com/{matches[0][0]}/refs/heads/{matches[0][1]}"
            try:
                response = await client.get(raw_url)
                response.raise_for_status()
            except httpx.TimeoutException:
                return f"Timeout encountered when retrieving resource: {raw_url}"

            file_lines = response.text.splitlines()
            selected_line = file_lines[line_number]

            return f"```{selected_line}\n```"

        async def get_lines(url_text: str) -> str:

            pattern = r"https://github.com/(.*)/blob/(.*)(#L(\d+)-L(\d+))"
            matches = re.findall(pattern, url_text)

            if len(matches) == 0:
                return ("Invalid github url. Please provide a valid github url with line information. "
                        "Example: 'https://github.com/my_repository/blob/main/file.txt#L409-L417'")

            start_line, end_line = int(matches[0][3]), int(matches[0][4])
            raw_url = f"https://raw.githubusercontent.com/{matches[0][0]}/refs/heads/{matches[0][1]}"
            try:
                response = await client.get(raw_url)
                response.raise_for_status()
            except httpx.TimeoutException:
                return f"Timeout encountered when retrieving resource: {raw_url}"

            file_lines = response.text.splitlines()
            selected_lines = file_lines[start_line:end_line]
            joined_selected_lines = "\n".join(selected_lines)

            return f"```{joined_selected_lines}\n```"

    group = FunctionGroup(config=config)

    group.add_function("get",
                       get,
                       description="Returns the text of a github file using a github url"
                       "starting with https://github.com and ending with a specific file.")
    group.add_function("get_line",
                       get_line,
                       description="Returns the text of a github file using a github url"
                       "starting with https://github.com and ending with a specific file"
                       "line reference. Examples of line references are #L409 and #L166.")
    group.add_function("get_lines",
                       get_lines,
                       description="Returns the text lines of a github file using a github url"
                       "starting with https://github.com and ending with a specific file"
                       "line reference.Examples of line references are #L409-L417 and #L166-L171.")

    yield group
