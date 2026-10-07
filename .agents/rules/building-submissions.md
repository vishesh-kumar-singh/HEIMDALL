---
trigger: always_on
---

1. Always while making a submission use the build package as instructed in the README.
2. While creating any new submission, don't make files from scratch, always use existing solution and copy it... make sure of how it uses dynamicaly number of GPUs and how exactly existing solutions parallelise.
3. Always build submission using build script, with small flag and test on train-sample folder.
4. Never hardcode the Cuda devices in the code... always use via export Cuda visible devices in the terminal before running main code, the code should do everything dynamically like other solutions do right now.