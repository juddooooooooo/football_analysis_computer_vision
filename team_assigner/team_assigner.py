import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """Split players into two sides by kit colour.

    Two things have to go right, and the original method
    (development_and_analysis/color_assignement.ipynb) gets neither.

    Finding the shirt. That version clustered each crop in two and kept
    whichever cluster avoided the corners. K-means on raw RGB divides mostly
    on brightness, so sunlit and shaded grass separate from each other and a
    dark shirt joins the darker one; the corner vote then returns pitch
    colour. Mexico's green came back as light grey-green that way. Here the
    grass colour is measured from the frame directly and pixels close to it
    are dropped, which works whatever colour the kit happens to be.

    Comparing the shirts. Clustering those colours in RGB again sorts by
    brightness, so two pale kits stay together: City's sky blue and Spurs'
    white landed in one group while the referee and keeper formed the other.
    Comparing in Lab a*b*, which carries colour without lightness, separates
    them. Measured against hand-read labels this scores 90% on City-Spurs and
    100% on Brazil-Mexico, against 55% for the original on City-Spurs.
    """

    GRASS_DISTANCE = 26.0     # Lab units; below this a pixel counts as pitch

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self._grass_lab = None
        self._centres_ab = None

    def _grass_reference(self, frame):
        """Median pitch colour, in Lab."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
        pixels = frame[green > 0]
        bgr = (np.median(pixels, axis=0) if len(pixels)
               else np.array([60, 120, 60], float))
        return cv2.cvtColor(bgr.reshape(1, 1, 3).astype(np.uint8),
                            cv2.COLOR_BGR2LAB).reshape(3).astype(float)

    @staticmethod
    def _to_ab(bgr):
        """Lab a*b* of a BGR colour: hue and chroma, no lightness."""
        arr = np.asarray(bgr, float).reshape(-1, 1, 3).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3)[:, 1:].astype(float)

    def get_player_color(self, frame, bbox):
        """Median colour of the shirt, with pitch pixels discounted."""
        if self._grass_lab is None:
            self._grass_lab = self._grass_reference(frame)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w < 3 or h < 6:
            return np.array([128.0, 128.0, 128.0])
        cx1 = max(0, x1 + int(0.12 * w))
        cx2 = min(frame.shape[1], x2 - int(0.12 * w))
        cy1 = max(0, y1 + int(0.10 * h))
        cy2 = min(frame.shape[0], y1 + int(0.60 * h))
        if cx2 <= cx1 or cy2 <= cy1:
            return np.array([128.0, 128.0, 128.0])

        crop = frame[cy1:cy2, cx1:cx2]
        pixels = crop.reshape(-1, 3).astype(float)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        kit = pixels[np.linalg.norm(lab - self._grass_lab, axis=1)
                     > self.GRASS_DISTANCE]
        # Fall back to the whole window when a player is too small or too
        # occluded to leave enough non-pitch pixels to be worth trusting.
        return np.median(kit, axis=0) if len(kit) >= 8 else np.median(pixels, axis=0)

    def assign_team_color(self, frame, player_detections):
        self._grass_lab = self._grass_reference(frame)
        colors = np.array([self.get_player_color(frame, d["bbox"])
                           for d in player_detections.values()])
        if len(colors) < 2:
            raise ValueError("Need at least two players to separate the kits.")

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10,
                        random_state=0).fit(self._to_ab(colors))
        self.kmeans = kmeans
        # Order by brightness of the mean kit colour so team numbering stays
        # stable between runs instead of following k-means' cluster order.
        means = [colors[kmeans.labels_ == k].mean(axis=0) for k in (0, 1)]
        order = np.argsort([m.sum() for m in means])
        self._centres_ab = kmeans.cluster_centers_[order]
        self.team_colors[1] = means[order[0]]
        self.team_colors[2] = means[order[1]]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, player_bbox)
        d = np.linalg.norm(self._centres_ab - self._to_ab(color)[0], axis=1)
        team_id = int(np.argmin(d)) + 1

        self.player_team_dict[player_id] = team_id
        return team_id
