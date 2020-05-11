TorchServe Developers Cookbook

TorchServe repo will have two kinds of customers:
TorchServe Users
TorchServe Contributors

TorchServe Users, who are interested in simply using TorchServe as an application to serve models, should install TorchServe using one of the methods documented in the serve/README.md document. We do not expect these users to make any changes to the source code of TorchServe and therefore these users can skip this section.

TorchServe Contributors are the developers who would change TorchServe source code and contribute to its maintenance and development. These changes could be small bug fixes, documentation changes or new features development. This section of the document is for developers to understand the development process for contributing to TorchServe. Our intention is to move relevant parts of this section to the CONTRIBUTING.md document under the TorchServe repo.

Before going through the Lifecycle of a Change, you will need to set a sane development environment for yourself. Please follow the section ‘Install TorchServe for Development’ in README.md to prepare your development workspace for TorchServe.

Lifecycle of a Change

User Story: As a developer, I would like to contribute a change to TorchServe, and while doing this, follow the development guidelines as prescribed for this project.

Identify the change
As a first step, you should identify the change that you want to make. There could be an existing issue for the change that you want to make, or if none exists, you should first create an issue for the change. The description for the issue should clearly state:
What do we want to solve
Why is this change needed
What is the definition of done
Exit criteria for this stage is one issue (close duplicate issues if you find them), which has enough information for someone to work on it. If you will work on this issue, you should assign the issue to yourself.

Work on the change

Issue is assigned to you
When you want to start working on a change, please make sure that the issue is assigned to you. This will be an indication to others that you are already picking up this issue and therefore will help avoid duplication of work.

Create a new development branch to make changes
You should create a separate branch for making your changes. You should create this new branch off the base branch that you would like to merge your changes to. For release v0.1.1, this should be the HEAD of the staging_0_1_1 branch. You should make sure that the base branch is updated in your local workspace by running this command:
 
git fetch --all

Then create a new branch with the name of the issue as the branch name using this command:
 
git checkout -b issue_333 origin/staging_0_1_1

Ensure that you can build locally
Before starting to make changes, you should test your local build environment. You can do this by running the following command and check if it succeeds:
 
scripts/install_from_src_ubuntu  #Change ubuntu to mac for OS X

Make your changes
Now, you are ready to make local changes. Once you want to test your changes, you should run the local build and test script again to make sure that there are no errors introduced by your changes.
 
scripts/install_from_src_ubuntu  #Change ubuntu to mac for OS X

Push your changes to run CI tests
Once you have done some basic testing for any regressions against your changes, you should push your branch to GitHub repo. This will automatically trigger a CI test run and the results should tell you whether your changes have caused any regressions against the HEAD of the base branch.
 
git push origin issue_333:issue_333

This should create a new branch on GitHub against which you should create a PR.

Create a new PR for your change
You should create a PR and link it to the issue. This will help you manage the lifecycle of the change through reviews, approvals and merge.


Developer checklist and steps for making a change:
Issue is assigned to you
Create a separate branch to work on
Run the script ‘scripts/install_from_src_ubuntu’
Create a linked PR for the issue



