if True:
    
    import sys
    sys.path.append (r'~/Desktop/Projects/Trading/cpp_utils/cpp_utils_v2/bin/Debug')
    
    from cpp_utils_v2 import *
    import numpy as np
    
    import os
    import sys
    import threading
    
    
    class OutputGrabber(object):
        """
        Class used to grab standard output or another stream.
        """
        escape_char = "\b"
    
        def __init__(self, stream=None, threaded=False):
            self.origstream = stream
            self.threaded = threaded
            if self.origstream is None:
                self.origstream = sys.stdout
            self.origstreamfd = self.origstream.fileno()
            self.capturedtext = ""
            # Create a pipe so the stream can be captured:
            self.pipe_out, self.pipe_in = os.pipe()
    
        def __enter__(self):
            self.start()
            return self
    
        def __exit__(self, type, value, traceback):
            self.stop()
    
        def start(self):
            """
            Start capturing the stream data.
            """
            self.capturedtext = ""
            # Save a copy of the stream:
            self.streamfd = os.dup(self.origstreamfd)
            # Replace the original stream with our write pipe:
            os.dup2(self.pipe_in, self.origstreamfd)
            if self.threaded:
                # Start thread that will read the stream:
                self.workerThread = threading.Thread(target=self.readOutput)
                self.workerThread.start()
                # Make sure that the thread is running and os.read() has executed:
                time.sleep(0.01)
    
        def stop(self):
            """
            Stop capturing the stream data and save the text in `capturedtext`.
            """
            # Print the escape character to make the readOutput method stop:
            self.origstream.write(self.escape_char)
            # Flush the stream to make sure all our data goes in before
            # the escape character:
            self.origstream.flush()
            if self.threaded:
                # wait until the thread finishes so we are sure that
                # we have until the last character:
                self.workerThread.join()
            else:
                self.readOutput()
            # Close the pipe:
            os.close(self.pipe_out)
            # Restore the original stream:
            os.dup2(self.streamfd, self.origstreamfd)
    
        def readOutput(self):
            """
            Read the stream data (one byte at a time)
            and save the text in `capturedtext`.
            """
            while True:
                data = os.read(self.pipe_out, 1)
                if not data or self.escape_char in data:
                    break
                self.capturedtext += data
                
    out = OutputGrabber()
    #out.start()
    
    a = boost_multi_array_2d_from_numpy_array(np.eye(5))
    
    l = nested_list_from_boost_multi_array_2d(a)
    
    print str(l)
    
    #out.stop ()