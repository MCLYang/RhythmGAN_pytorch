# Use an official Node.js runtime as the base image
FROM node:14

# Set the working directory in the container to /app
WORKDIR /app

# Copy package.json and package-lock.json into the working directory
COPY package*.json ./

# Install the http-server package
RUN npm install --global http-server

# Copy the rest of the frontend code into the container
COPY . .

# Make the server accessible on port 8000
EXPOSE 8000

# Start the server
CMD [ "http-server", "-p", "8000" ]