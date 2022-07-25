#!/usr/bin/env bash

set -e # exit immediately after any error

echo "
#########################################################################################
### STEP : Initialization
###
### - Setup environment variables for the pipeline
### - Conditional tasks :
###     --> release/hotfix : create GitHub release
#########################################################################################
"

#
# 0. Import functions
#
# . "${WORKDIR}/scripts/pipeline/functions/github.sh"
. "${WORKDIR}/scripts/pipeline/functions/notification.sh"

#
# 1. Set possible versions variables
#

# CURRENT_RELEASE=$(gh release -R "${GITHUB_REPOSITORY}" view --json name | jq -r .name)
# SPLIT_RELEASE=(${CURRENT_RELEASE//./ })
# MAJOR=${SPLIT_RELEASE[0]}
# MINOR=${SPLIT_RELEASE[1]}
# POINT=${SPLIT_RELEASE[2]}

# LONG_CURRENT_VERSION="${CURRENT_RELEASE}-${GIT_SHORT_COMMIT}"
# LONG_CURRENT_VERSION_WITH_TIMESTAMP="${LONG_CURRENT_VERSION}-$(date +%s)"
# NEXT_SHORT_RELEASE_VERSION="${MAJOR}.$(expr ${MINOR} + 1).0"
# NEXT_LONG_RELEASE_VERSION="${NEXT_SHORT_RELEASE_VERSION}-${GIT_SHORT_COMMIT}"
# NEXT_SHORT_HOTFIX_VERSION="${MAJOR}.${MINOR}.$(expr ${POINT} + 1)"

# echo ">>> The possible new versions are:
#         feature:          ${LONG_CURRENT_VERSION_WITH_TIMESTAMP}
#         deploy:           ${LONG_CURRENT_VERSION}
#         develop:          ${NEXT_LONG_RELEASE_VERSION}
#         release:          ${NEXT_SHORT_RELEASE_VERSION}
#         hotfix:           ${NEXT_SHORT_HOTFIX_VERSION}
# "

#
# 2. Declare environment variables with default values
#
# export NEW_VERSION="${LONG_CURRENT_VERSION_WITH_TIMESTAMP}"
export IS_DEVELOP_BRANCH="N"

#
# 3. Updates environment variables based on branch pattern
#
case "${GIT_BRANCH}" in
  "develop")
    # NEW_VERSION="${NEXT_LONG_RELEASE_VERSION}"
    IS_DEVELOP_BRANCH="Y"
    ;;
  *)
    ;;
esac

#
# 4. Additional environment variables
#
export IMAGE_TAG="${QUAY_HOSTNAME}/${QUAY_REPOSITORY}:${GIT_SHORT_COMMIT}"

# export THIRD_PARTY_IMAGE_SERVICE_MESH_INTEGRATION="${QUAY_HOSTNAME}/servicemesh/integration:${SERVICE_MESH_VERSION}"
# export THIRD_PARTY_IMAGE_SERVICE_MESH_BEIPR1="${QUAY_HOSTNAME}/servicemesh/beipr1:${SERVICE_MESH_VERSION}"

#
# 5. Initialization output
#
echo ">>> Initialization output :"
env

#
# 6. Send slack notification
#
send_slack_message "Repo: ${GITHUB_REPOSITORY} Branch: ${GIT_BRANCH}" "#87ceeb" "Build started with version ${GIT_SHORT_COMMIT} by ${COMMIT_AUTHOR} (pattern: ${BRANCH_PATTERN})"

#
# 7. Create GitHub release (release/hotfix only)
#
# if [ "${IS_RELEASE_HOTFIX_BRANCH}" == "Y" ]; then
#   run_create_github_release
# fi