ARG BASE_IMAGE
FROM --platform=linux/amd64 ${BASE_IMAGE}

# Install patch tool
RUN apt-get update && \
    apt-get install -y patch && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the main patch file
COPY patch.txt /patch.txt
WORKDIR /

# Set shell option to exit immediately if a command exits with non-zero status
SHELL ["/bin/bash", "-e", "-c"]

# Apply patch to the installed packages
RUN if [ -s /patch.txt ]; then \
      PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
      echo "Python packages located at: $PYTHON_SITE_PACKAGES" && \
      echo "Applying patch to packages in $PYTHON_SITE_PACKAGES" && \
      cd $PYTHON_SITE_PACKAGES && \
      if patch -p1 --dry-run < /patch.txt > /dev/null 2>&1; then \
        patch -p1 < /patch.txt || { echo "Failed to apply patch!"; exit 1; }; \
        echo "✓ Successfully applied patch.txt"; \
      else \
        echo "⚠️ Warning: Trying with -p0 instead..."; \
        if patch -p0 --dry-run < /patch.txt > /dev/null 2>&1; then \
          patch -p0 < /patch.txt || { echo "Failed to apply patch!"; exit 1; }; \
          echo "✓ Successfully applied patch.txt with -p0"; \
        else \
          echo "❌ Error: patch.txt cannot be applied cleanly."; \
          exit 1; \
        fi; \
      fi; \
    else \
      echo "No patch to apply (patch.txt is empty)"; \
    fi

WORKDIR /app/tensorrt_llm

RUN pip install nixl
