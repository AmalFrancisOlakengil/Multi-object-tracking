
import numpy as np

class CentroidTracker:

    # max_disappeared is the maximum amount of frames allowed where the object is not found
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}  # Stores {ID : (x,y)} where (x,y) is the centroid
        self.disappeared = {}  # Stores {ID : number of consecutive frames missed}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # Assign a new ID to a new object
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0 # initialize to 0
        self.next_object_id += 1 # for the next new object

    def deregister(self, object_id):
        # Remove the object from tracking
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # here rects is a list of bouding box vectors 
        # If no objects are detected in the current frame
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate centroids for current frame detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # If we aren't tracking anything yet, register all input centroids
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Grab the set of object IDs and their corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distances between history centroids and new input centroids
            # Rows = existing objects, Columns = new detections
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find the smallest value in each row, then sort row indexes based on minimum values
            rows = D.min(axis=1).argsort()
            # Find the smallest value in each column, then sort using the row map
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            # Loop over the combination of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Check if any objects have disappeared or if there are new detections
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Handle disappeared objects
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Handle brand new detections
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects