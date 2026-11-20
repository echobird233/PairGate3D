
class Object:
    '''
    Container for one per-frame tracking state, including predictions, updates, scores, and
    features.
    '''
    def __init__(self):
        '''Initialize the object state and configure the resources required by this component.'''
        self.updated_state=None
        self.predicted_state=None
        self.detected_state=None
        self.updated_covariance=None
        self.predicted_covariance=None
        self.prediction_score=None
        self.score=None
        self.features=None
        # Diagnostic provenance independent of updated_state because filtering
        # may later copy a prediction into that field.
        self.state_source=None
        self.detector_score=None
        # P0-A source-aware diagnostic fields. These are metadata only and are
        # deliberately excluded from all tracker decisions.
        self.gap_age=None
        self.track_age=None
        self.num_updates=None
        self.init_score=None
        self.legitimacy=None
        self.last_detection_frame=None
        self.elapsed_since_detection=None
        self.sensor_depth=None
        self.emit_eligible=True
        self.emit_suppression_reason=None
        self.prediction_velocity_residual=None
        self.recovery_source=None
        self.emit_eligible=True
        self.emit_suppression_reason=None
