#!/usr/bin/env bash

#########################################################
### This file gathers functions related to notifications.
#########################################################

#
# Main function to send slack message in #raas-ci.
#
function send_slack_message {
  local title="${1}"
  local color="${2}"
  local message="${3}"
  local channel="#fapiao-ci"
  local footer="Click <https://codebuild.cnqr.delivery|here> and select 'fapiao' to log in AWS build account, then click <${CODEBUILD_BUILD_URL}|here> to access your build logs."

  echo ">>> Send notification in Slack channel ${channel} :
          message: ${message}
  "

  "${WORKDIR}"/secure-pipeline-scripts/notification/slack.sh "${channel}" "${title}" "${color}" "${message}" "${footer}"
}