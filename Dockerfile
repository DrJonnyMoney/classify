# Use the Kubeflow Code-Server Python image as base
FROM kubeflownotebookswg/codeserver-python:latest

# Switch to root to install packages
USER root

# Install dependencies
RUN pip install --no-cache-dir torch torchvision pillow flask gunicorn

# Create app directories
RUN mkdir -p /tmp_home/jovyan/webapp/templates

# Copy application files to the temporary home directory
COPY app.py /tmp_home/jovyan/webapp/
COPY templates/ /tmp_home/jovyan/webapp/templates/

# Set correct permissions
RUN chown -R ${NB_USER}:${NB_GID} /tmp_home/jovyan/webapp

# Remove the code-server service to prevent it from starting
RUN rm -f /etc/services.d/code-server/run || true

# Create flask service directory
RUN mkdir -p /etc/services.d/flask

# Create the run script for Flask
COPY flask-run /etc/services.d/flask/run
RUN chmod 755 /etc/services.d/flask/run && \
    chown ${NB_USER}:${NB_GID} /etc/services.d/flask/run

# Expose port 8888 (standard for Kubeflow notebooks)
EXPOSE 8888

# Switch back to non-root user
USER $NB_UID

# Use the built-in s6-overlay entrypoint
ENTRYPOINT ["/init"]
