FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the project and install it, so the container gets exactly the same
# dependency set as a `pip install reqreate[app]` on a user's machine.
COPY . .
RUN pip install --no-cache-dir ".[app]"

# Expose Streamlit port
EXPOSE 7860

CMD ["reqreate", "app", "--port", "7860", "--address", "0.0.0.0", "--no-browser"]
