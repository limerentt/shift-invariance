#!/bin/bash

# Clone the latexrun tool if it doesn't exist
if [ ! -d "latexrun" ]; then
  echo "Cloning latexrun..."
  git clone https://github.com/aclements/latexrun.git
fi

echo "Configuration complete. Run 'make' to build the thesis."
