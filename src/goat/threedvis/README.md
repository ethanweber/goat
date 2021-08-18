# threedvis (3D vis)

The 3D visualizer contains most code from [MeshCat](https://github.com/rdeits/meshcat-python). We've begun porting the [current interface](https://github.com/rdeits/meshcat) to React JS. The code consists of two components:

- **meschat** - The server-side code that handles communication with the three.js client.
- **puppy** - The client-side three.js wrapper built with React JS.

```
# start the server-side websocket handler for TCP routing
cd meshcat
python run_start_server.py

# start the React JS, client-side visualizer
cd puppy
npm install
npm start

# navigate to the visualizer
# in the following example, visiongpu07.csail.mit.edu:8051 is where you ran run_start_server.py
http://localhost:3000/visiongpu07.csail.mit.edu:8051
```