"""Helper code for wrapping the boto3 MTurk API.
"""


class MTurkWrapper(object):
    """This class is meant to make using the boto3 API more efficient for research projects.
    """

    def __init__(self, database_name):
        """
        :param database_name: keep track of HITs with associated meta-data, defined by the user, in a local database
        """
        self.database_name = database_name

    def submit_external_url(self, external_url):
        """Submit an external_url.
        """
        pass

    # TODO(ethan): reference Scaling ADE Looper() class and write this class
