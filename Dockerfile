# 1. Start with a tiny version of Linux that has Conda installed
FROM continuumio/miniconda3

# 2. Set the "home base" folder inside the container
WORKDIR /app

# 3. Copy your shopping list into the container
COPY environment.yml .

# 4. Tell the container to install everything on that list
RUN conda env update -n base -f environment.yml

# 5. Copy your Python code (your .py files) into the container
COPY . .

# 6. Tell the container what to do when it starts
CMD ["python", "hi.py"]