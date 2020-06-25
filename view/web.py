"""Helper code to visualize images quickly and efficiently in the web.

Examples:
    https://gist.github.com/ethanweber/79dd0a0a1341c243e26bae3772cd4505
"""


def get_html_from_image_urls(image_urls):
    html_str = """"""
    for image_url in image_urls:
        html_str += """<img src="{}" height="100px" width="100px">""".format(image_url)
    return html_str
