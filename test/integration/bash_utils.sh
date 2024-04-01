#!/bin/bash

bash cmd; rm -rf module.nemd.* tmp.data polymer_builder-driver.log signac_* polymer_builder.in; git add .; git commit -a -m "fix $(basename "$PWD")"; git push origin master
