    def Angle(self, img, reference_point, feature1, feature2):
        reference = self.lmList[reference_point][1:]

        point1 = np.array(self.lmList[feature1][1:])  # Convert to NumPy array
        point2 = np.array([self.lmList[feature2][1], 0])  # Convert to NumPy array
        p1_ref = point1 - reference
        p2_ref = point2 - reference
        cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        degree = int(180 / np.pi) * theta

        ###

        multiplier = -1
        cv2.line(img, self.lmList[23][1:], self.lmList[11][1:], (102, 204, 255), 4, cv2.LINE_AA)
        cv2.line(img, self.lmList[11][1:], self.lmList[13][1:], (102, 204, 255), 4, cv2.LINE_AA)
        cv2.circle(img, self.lmList[11][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, self.lmList[13][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, self.lmList[23][1:], 7, (0, 0, 255), -1, cv2.LINE_AA)


        return int(degree)


