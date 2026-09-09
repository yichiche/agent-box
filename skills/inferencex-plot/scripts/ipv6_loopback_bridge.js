// Bridge [::1]:PORT -> 127.0.0.1:PORT.
//
// Vite binds a single address. Bound to 127.0.0.1 it is invisible over IPv6,
// and `localhost` resolves to ::1 first on this host, so an editor's port
// forwarder that does not fall back to IPv4 fails with "connection reset".
// Binding 0.0.0.0 would fix it by exposing the port to the whole LAN; this
// keeps the server loopback-only instead.
const net = require('node:net');
const port = Number(process.argv[2] || 5173);

net.createServer((from) => {
  const to = net.connect(port, '127.0.0.1');
  from.on('error', () => to.destroy());
  to.on('error', () => from.destroy());
  from.pipe(to);
  to.pipe(from);
}).listen(port, '::1', () => console.log(`bridging [::1]:${port} -> 127.0.0.1:${port}`));
