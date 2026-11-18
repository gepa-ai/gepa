"""Blog listing helpers: Community section + sidebar post lists."""

from __future__ import annotations

from mkdocs.plugins import event_priority
from mkdocs.structure.files import InclusionLevel
from mkdocs.structure.nav import Section

try:
    from material.plugins.blog.structure import Post
except ImportError:  # pragma: no cover
    Post = ()  # type: ignore


def _blog_plugin(config):
    plugin = config.plugins.get("blog")
    if plugin is not None and hasattr(plugin, "blog"):
        return plugin
    for candidate in config.plugins.values():
        if hasattr(candidate, "blog") and hasattr(candidate.blog, "posts"):
            return candidate
    return None


def _src_uri(item) -> str | None:
    file = getattr(item, "file", None)
    return getattr(file, "src_uri", None)


def _page_for(files, src_uri: str):
    for f in files:
        if f.src_uri == src_uri and f.page is not None:
            return f.page
    return None


def on_page_context(context, page, config, nav):
    if page.file.src_uri != "blog/community/index.md":
        return

    plugin = _blog_plugin(config)
    if plugin is None:
        context["posts"] = []
        context["pagination"] = None
        return

    separator = plugin.config.post_excerpt_separator
    excerpts = []
    for post in plugin.blog.posts:
        if not post.meta.get("guest"):
            continue
        excerpt = getattr(post, "excerpt", None)
        if excerpt is None:
            continue
        excerpt.render(page, separator)
        excerpts.append(excerpt)

    context["posts"] = excerpts
    context["pagination"] = None


@event_priority(-100)
def on_nav(nav, config, files):
    """Under Blog: index, team posts, Community (landing + guest posts), then Archive.

    With ``navigation.indexes``, the first child of a section becomes the
    section link and is omitted from the list — so the blog index and the
    community landing page must come first.
    """
    plugin = _blog_plugin(config)
    if plugin is None or not hasattr(plugin, "blog"):
        return nav

    blog = plugin.blog
    if blog is None or blog.parent is None:
        return nav

    parent = blog.parent
    if not isinstance(parent, Section):
        return nav

    team = [p for p in blog.posts if not p.meta.get("guest")]
    guest = [p for p in blog.posts if p.meta.get("guest")]

    for post in team + guest:
        post.file.inclusion = InclusionLevel.INCLUDED

    # Prefer the instance already in the nav tree (same object MkDocs will render).
    community_landing = next(
        (child for child in parent.children if _src_uri(child) == "blog/community/index.md"),
        None,
    )
    if community_landing is None:
        community_landing = _page_for(files, "blog/community/index.md")
    if community_landing is not None:
        community_landing.file.inclusion = InclusionLevel.INCLUDED

    kept = []
    for child in list(parent.children):
        uri = _src_uri(child)
        if child is blog or uri == "blog/index.md" or uri == "blog/community/index.md":
            continue
        if Post and isinstance(child, Post):
            continue
        if isinstance(child, Section) and child.title == "Community":
            continue
        kept.append(child)

    new_children = [blog, *team]

    if guest:
        community_children = [community_landing, *guest] if community_landing is not None else list(guest)
        community = Section("Community", community_children)
        community.parent = parent
        for child in community.children:
            child.parent = community
        new_children.append(community)

    new_children.extend(kept)
    parent.children = new_children
    for child in parent.children:
        child.parent = parent

    return nav
