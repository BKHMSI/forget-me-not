import os
import time

from http.server import SimpleHTTPRequestHandler, BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from evaluate import Evaluate

HOST_NAME = ""  
PORT_NUMBER = 8080
web_dir = "web"
fmn = Evaluate()

# This class contains methods to handle our requests to different URIs in the app
class MyHandler(SimpleHTTPRequestHandler):

    # def __init__(self, request, client_address, server):
    #     SimpleHTTPRequestHandler.__init__(self, request, client_address, server)

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
 
    # Check the URI of the request to serve the proper content.
    def do_GET(self):
        if "query" in self.path:
        	# If URI contains URLToTriggerGetRequestHandler, execute the python script that corresponds to it and get that data
            # whatever we send to "respond" as an argument will be sent back to client
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            
            query    = params["query"][0]
            content  = fmn.main(params["video"], query)
            self.respond(content) # we can retrieve response within this scope and then pass info to self.respond
        else:

            if self.path == "/":
                self.path = "/index.html"
            try:
                sendReply = False
                if self.path.endswith(".html"):
                    mimetype='text/html'
                    sendReply = True
                if self.path.endswith(".jpg"):
                    mimetype='image/jpg'
                    sendReply = True
                if self.path.endswith(".png"):
                    mimetype='image/png'
                    sendReply = True
                if self.path.endswith(".gif"):
                    mimetype='image/gif'
                    sendReply = True
                if self.path.endswith(".mp4"):
                    mimetype='video/mp4'
                    sendReply = True
                if self.path.endswith(".js"):
                    mimetype='application/javascript'
                    sendReply = True
                if self.path.endswith(".css"):
                    mimetype='text/css'
                    sendReply = True

                if sendReply == True:
                    #Open the static file requested and send it
                    self.send_response(200)
                    self.send_header('Content-type', mimetype)
                    self.end_headers()
                    if self.path.endswith('.png') or self.path.endswith('.gif') or self.path.endswith('.jpg') or self.path.endswith(".mp4"):
                        self.wfile.write(self.load_binary(web_dir+self.path))
                    else:
                        f = open(web_dir+self.path) 
                        self.wfile.write(bytes(f.read(), "utf-8"))
                        f.close()

            except IOError:
                self.send_error(404,'File Not Found: %s' % self.path)
 
    def load_binary(self, file):
        with open(file, 'rb') as file:
            return file.read()
        
    def do_POST(self):
        if "query" in self.path:
            return

    def handle_http(self, data):
        self.send_response(200)
        # set the data type for the response header. In this case it will be json.
        # setting these headers is important for the browser to know what 	to do with
        # the response. Browsers can be very picky this way.
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        return bytes(data, "utf-8")
 
     # store response for delivery back to client. This is good to do so
     # the user has a way of knowing what the server's response was.
    def respond(self, data):
        response = self.handle_http(data)
        self.wfile.write(response)
 
# This is the main method that will fire off the server. 
if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
