#!/usr/bin/python3
r"""
This bot uses the TextBlob module to do nlp analysis and correct accordingly.

It will retrieve information on which pages might need changes either from
an XML dump or a text file, or only change a single page.

These command line parameters can be used to specify which pages to work on:

&params;

Furthermore, the following command line parameters are supported:

-mysqlquery       Retrieve information from a local database mirror.
                  If no query specified, bot searches for pages with
                  given replacements.

-xml              Retrieve information from a local XML dump
                  (pages-articles or pages-meta-current, see
                  https://dumps.wikimedia.org). Argument can also
                  be given as "-xml:filename".

                  -xmlstart:Article).

-addcat:cat_name  Adds "cat_name" category to every altered page.

-excepttitle:XYZ  Skip pages with titles that contain XYZ. If the -regex
                  argument is given, XYZ will be regarded as a regular
                  expression.

-requiretitle:XYZ Only do pages with titles that contain XYZ. If the -regex
                  argument is given, XYZ will be regarded as a regular
                  expression.

-excepttext:XYZ   Skip pages which contain the text XYZ. If the -regex
                  argument is given, XYZ will be regarded as a regular
                  expression.

-exceptinside:XYZ Skip occurrences of the to-be-replaced text which lie
                  within XYZ. If the -regex argument is given, XYZ will be
                  regarded as a regular expression.

-exceptinsidetag:XYZ Skip occurrences of the to-be-replaced text which lie
                  within an XYZ tag.

-summary:XYZ      Set the summary message text for the edit to XYZ, bypassing
                  the predefined message texts with original and replacements
                  inserted. To add the replacements to your summary use the
                  %(description)s placeholder, for example:
                  -summary:"Bot operated replacement: %(description)s"
                  Can't be used with -automaticsummary.

-automaticsummary Uses an automatic summary for all replacements which don't
                  have a summary defined. Can't be used with -summary.

-sleep:123        If you use -fix you can check multiple regex at the same time
                  in every page. This can lead to a great waste of CPU because
                  the bot will check every regex without waiting using all the
                  resources. This will slow it down between a regex and another
                  in order not to waste too much CPU.

-always           Don't prompt you for each replacement

-quiet            Don't prompt a message if a page keeps unchanged

-nopreload        Do not preload pages. Useful if disabled on a wiki.

-fullsummary      Use one large summary for all command line replacements.

Please type "python pwb.py nlp_correct -help | more" if you can't read
the top of the help.
"""
#
# Substantial portions of this code are butchered and derived from the replace.py script, which is (C) Pywikibot team, 2004-2022, which has been modified and is distributed under the terms of the MIT license.
# Subsequent code is licensed per GNU 3.0
#

try:
    import textblob
except ModuleNotFoundError:
    textblob_imported = False
else:
    textblob_imported = True
import codecs
import re
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Optional

import pywikibot
from pywikibot import editor, fixes, i18n, pagegenerators, textlib
from pywikibot.backports import Dict, Generator, List, Pattern, Tuple
from pywikibot.bot import ExistingPageBot, SingleSiteBot
from pywikibot.exceptions import InvalidPageError, NoPageError
from pywikibot.tools import chars


# This is required for the text that is shown when you run this script
# with the parameter -help.
docuReplacements = {
    '&params;': pagegenerators.parameterHelp,
    '&fixes-help;': fixes.parameter_help,
}


def _get_text_exceptions(exceptions):
    """Get exceptions on text (inside exceptions)."""
    return exceptions.get('inside-tags', []) + exceptions.get('inside', [])


class ReplacementBase:

    """The replacement instructions."""

    def __init__(
        self,
        old,
        new,
        edit_summary=None,
        default_summary=True
    ) -> None:
        """Create a basic replacement instance."""
        self.old = old
        self.old_regex = None
        self.new = new
        self._edit_summary = edit_summary
        self.default_summary = default_summary

    @property
    def edit_summary(self) -> str:
        """Return the edit summary for this fix."""
        return self._edit_summary

    @property
    def description(self) -> str:
        """Description of the changes that this replacement applies.

        This description is used as the default summary of the replacement. If
        you do not specify an edit summary on the command line or in some other
        way, whenever you apply this replacement to a page and submit the
        changes to the MediaWiki server, the edit summary includes the
        descriptions of each replacement that you applied to the page.
        """
        return '-{} +{}'.format(self.old, self.new)

    @property
    def container(self):
        """Container object which contains this replacement.

        A container object is an object that groups one or more replacements
        together and provides some properties that are common to all of them.
        For example, containers may define a common name for a group of
        replacements, or a common edit summary.

        Container objects must have a "name" attribute.
        """
        return None

class Replacement(ReplacementBase):

    """A single replacement with it's own data."""

    def __init__(self, old, new, use_regex=None, exceptions=None,
                 case_insensitive=None, edit_summary=None,
                 default_summary=True) -> None:
        """Create a single replacement entry unrelated to a fix."""
        super().__init__(old, new, edit_summary, default_summary)
        self._use_regex = use_regex
        self.exceptions = exceptions
        self._case_insensitive = case_insensitive

    def get_inside_exceptions(self):
        """Get exceptions on text (inside exceptions)."""
        return _get_text_exceptions(self.exceptions or {})


class ReplacementList(list):

    """
    A list of replacements which all share some properties.

    The shared properties are:
    * use_regex
    * exceptions
    * case_insensitive

    Each entry in this list should be a ReplacementListEntry. The exceptions
    are compiled only once.
    """

    def __init__(self, use_regex, exceptions, case_insensitive, edit_summary,
                 name) -> None:
        """Create a fix list which can contain multiple replacements."""
        super().__init__()
        self.use_regex = use_regex
        self._exceptions = exceptions
        self.exceptions = None
        self.case_insensitive = case_insensitive
        self.edit_summary = edit_summary
        self.name = name

    def _compile_exceptions(self, use_regex, flags) -> None:
        """Compile the exceptions if not already done."""
        if not self.exceptions and self._exceptions is not None:
            self.exceptions = dict(self._exceptions)
            precompile_exceptions(self.exceptions, use_regex, flags)


class ReplacementListEntry(ReplacementBase):

    """A replacement entry for ReplacementList."""

    def __init__(self, old, new, fix_set, edit_summary=None,
                 default_summary=True) -> None:
        """Create a replacement entry inside a fix set."""
        super().__init__(old, new, edit_summary, default_summary)
        self.fix_set = fix_set

    @property
    def exceptions(self):
        """Return the exceptions of the fix set."""
        return self.fix_set.exceptions

    @property
    def edit_summary(self):
        """Return this entry's edit summary or the fix's summary."""
        if self._edit_summary is None:
            return self.fix_set.edit_summary
        return self._edit_summary

    @property
    def container(self):
        """Container object which contains this replacement.

        A container object is an object that groups one or more replacements
        together and provides some properties that are common to all of them.
        For example, containers may define a common name for a group of
        replacements, or a common edit summary.

        Container objects must have a "name" attribute.
        """
        return self.fix_set

    def _compile(self, use_regex, flags) -> None:
        """Compile the search regex and the fix's exceptions."""
        super()._compile(use_regex, flags)
        self.fix_set._compile_exceptions(use_regex, flags)

    def get_inside_exceptions(self):
        """Get exceptions on text (inside exceptions)."""
        return _get_text_exceptions(self.fix_set.exceptions or {})


class XmlDumpReplacePageGenerator:

    """
    Iterator that will yield Pages that might contain text to replace.

    These pages will be retrieved from a local XML dump file.

    :param xmlFilename: The dump's path, either absolute or relative
    :param xmlStart: Skip all articles in the dump before this one
    :param replacements: A list of 2-tuples of original text (as a
        compiled regular expression) and replacement text (as a string).
    :param exceptions: A dictionary which defines when to ignore an
        occurrence. See docu of the ReplaceRobot initializer below.
    :type exceptions: dict
    """

    def __init__(self,
                 xmlFilename: str,
                 xmlStart: str,
                 replacements: List[Tuple[Any, str]],
                 exceptions: Dict[str, Any],
                 site) -> None:
        """Initializer."""
        self.xmlFilename = xmlFilename
        self.replacements = replacements
        self.exceptions = exceptions
        self.xmlStart = xmlStart
        self.skipping = bool(xmlStart)

        self.excsInside = []
        if 'inside-tags' in self.exceptions:
            self.excsInside += self.exceptions['inside-tags']
        if 'inside' in self.exceptions:
            self.excsInside += self.exceptions['inside']
        from pywikibot import xmlreader
        if site:
            self.site = site
        else:
            self.site = pywikibot.Site()
        dump = xmlreader.XmlDump(self.xmlFilename, on_error=pywikibot.error)
        self.parser = dump.parse()

    def __iter__(self):
        """Iterator method."""
        try:
            for entry in self.parser:
                if self.skipping:
                    if entry.title != self.xmlStart:
                        continue
                    self.skipping = False
                if self.isTitleExcepted(entry.title) \
                        or self.isTextExcepted(entry.text):
                    continue
                new_text = entry.text
                for replacement in self.replacements:
                    # This doesn't do an actual replacement but just
                    # checks if at least one does apply
                    new_text = textlib.replaceExcept(
                        new_text, replacement.old_regex, replacement.new,
                        self.excsInside + replacement.get_inside_exceptions(),
                        site=self.site)
                if new_text != entry.text:
                    yield pywikibot.Page(self.site, entry.title)

        except KeyboardInterrupt:
            with suppress(NameError):
                if not self.skipping:
                    pywikibot.output(
                        'To resume, use "-xmlstart:{}" on the command line.'
                        .format(entry.title))

    def isTitleExcepted(self, title) -> bool:
        """Return True if one of the exceptions applies for the given title."""
        if 'title' in self.exceptions:
            for exc in self.exceptions['title']:
                if exc.search(title):
                    return True
        if 'require-title' in self.exceptions:
            for req in self.exceptions['require-title']:
                if not req.search(title):  # if not all requirements are met:
                    return True

        return False

    def isTextExcepted(self, text) -> bool:
        """Return True if one of the exceptions applies for the given text."""
        if 'text-contains' in self.exceptions:
            return any(exc.search(text)
                       for exc in self.exceptions['text-contains'])
        return False


class ReplaceRobot(SingleSiteBot, ExistingPageBot):

    """A bot that does nlp text replacement

    :param generator: generator that yields Page objects
    :type generator: generator
    :keyword addcat: category to be added to every page touched
    :type addcat: pywikibot.Category or str or None
    :keyword sleep: slow down between processing multiple regexes
    :type sleep: int
    :keyword summary: Set the summary message text bypassing the default
    :type summary: str
    :keyword always: the user won't be prompted before changes are made
    :type keyword: bool
    :keyword site: Site the bot is working on.
    .. warning::
       - Be careful with `recursive` parameter, this might lead to an
         infinite loop.
       - `site` parameter should be passed to constructor.
         Otherwise the bot takes the current site and warns the operator
         about the missing site
    """

    def __init__(self, generator,
                 replacements: List[Tuple[Any, str]],
                 exceptions: Optional[Dict[str, Any]] = None,
                 **kwargs) -> None:
        """Initializer."""
        self.available_options.update({
            'addcat': None,
            'quiet': False,
            'sleep': 0.0,
            'summary': None,
        })
        super().__init__(generator=generator, **kwargs)

        self.exceptions = exceptions or {}

        if self.opt.addcat and isinstance(self.opt.addcat, str):
            self.opt.addcat = pywikibot.Category(self.site, self.opt.addcat)

    def isTitleExcepted(self, title, exceptions=None) -> bool:
        """Return True if one of the exceptions applies for the given title."""
        if exceptions is None:
            exceptions = self.exceptions
        if 'title' in exceptions:
            for exc in exceptions['title']:
                if exc.search(title):
                    return True
        if 'require-title' in exceptions:
            for req in exceptions['require-title']:
                if not req.search(title):
                    return True
        return False

    def isTextExcepted(self, original_text) -> bool:
        """Return True iff one of the exceptions applies for the given text."""
        if 'text-contains' in self.exceptions:
            return any(exc.search(original_text)
                       for exc in self.exceptions['text-contains'])
        return False

    def apply_replacements(self, original_text, applied, page=None):
        """
        Apply all replacements to the given text.

        :rtype: str, set
        """
        if page is None:
            pywikibot.warn(
                'You must pass the target page as the "page" parameter to '
                'apply_replacements().', DeprecationWarning, stacklevel=2)
        old_text = original_text
        exceptions = _get_text_exceptions(self.exceptions)
        skipped_containers = set()
        if self.opt.sleep:
            pywikibot.sleep(self.opt.sleep)
        new_text = ""
        for line in old_text.split("\n"):
            disallowed_chars = set("#$%()*+<=>@[]^_`{|}~")
            if set(disallowed_chars) - set(line) != disallowed_chars:
                #pywikibot.output('"' + line + '" has disallowed characters, skipping.')
                new_text += line
            else:
                new_text += str(textblob.TextBlob(line).correct())
            new_text += "\n"

        return new_text[:-1]

    def generate_summary(self, applied_replacements):
        """Generate a summary message for the replacements."""
        # all replacements which are merged into the default message
        default_summaries = set()
        # all message parts
        summary_messages = set()
        summary_messages = sorted(summary_messages)
        if default_summaries:
            if self.opt.summary:
                msg = self.opt.summary
            else:
                msg = i18n.twtranslate(self.site, 'replace-replacing')
            comma = self.site.mediawiki_message('comma-separator')
            default_summary = comma.join(
                '-{} +{}'.format(*default_summary)
                for default_summary in default_summaries)
            desc = {'description': ' ({})'.format(default_summary)}
            summary_messages.insert(0, msg % desc)

        semicolon = self.site.mediawiki_message('semicolon-separator')
        return semicolon.join(summary_messages)

    def skip_page(self, page):
        """Check whether treat should be skipped for the page."""
        if super().skip_page(page):
            return True

        if self.isTitleExcepted(page.title()):
            pywikibot.warning(
                'Skipping {} because the title is on the exceptions list.'
                .format(page))
            return True

        if not page.has_permission():
            pywikibot.warning("You can't edit page {}".format(page))
            return True

        return False

    def treat(self, page) -> None:
        """Work on each page retrieved from generator."""
        try:
            original_text = page.text
        except InvalidPageError as e:
            pywikibot.error(e)
            return
        applied = set()
        new_text = original_text
        last_text = None
        context = 0
        while True:
            if self.isTextExcepted(new_text):
                pywikibot.output('Skipping {} because it contains text '
                                 'that is on the exceptions list.'
                                 .format(page))
                return

            last_text = new_text
            new_text = self.apply_replacements(last_text, applied, page)

            if new_text == original_text:
                if not self.opt.quiet:
                    pywikibot.output('No changes were necessary in '
                                     + page.title(as_link=True))
                return

            if self.opt.addcat:
                # Fetch only categories in wikitext, otherwise the others
                # will be explicitly added.
                cats = textlib.getCategoryLinks(new_text, site=page.site)
                if self.opt.addcat not in cats:
                    cats.append(self.opt.addcat)
                    new_text = textlib.replaceCategoryLinks(new_text,
                                                            cats,
                                                            site=page.site)
            # Show the title of the page we're working on.
            # Highlight the title in purple.
            self.current_page = page
            pywikibot.showDiff(original_text, new_text, context=context)
            if self.opt.always:
                break

            choice = pywikibot.input_choice(
                'Do you want to accept these changes?',
                [('Yes', 'y'), ('No', 'n'), ('Edit original', 'e'),
                 ('edit Latest', 'l'), ('open in Browser', 'b'),
                 ('More context', 'm'), ('All', 'a')],
                default='N')
            if choice == 'm':
                context = context * 3 if context else 3
                continue
            if choice in ('e', 'l'):
                text_editor = editor.TextEditor()
                edit_text = original_text if choice == 'e' else new_text
                as_edited = text_editor.edit(edit_text)
                # if user didn't press Cancel
                if as_edited and as_edited != new_text:
                    new_text = as_edited
                    if choice == 'l':
                        # prevent changes from being applied again
                        last_text = new_text
                continue
            if choice == 'b':
                pywikibot.bot.open_webbrowser(page)
                try:
                    original_text = page.get(get_redirect=True, force=True)
                except NoPageError:
                    pywikibot.output('Page {} has been deleted.'
                                     .format(page.title()))
                    break
                new_text = original_text
                last_text = None
                continue
            if choice == 'a':
                self.opt.always = True
            if choice == 'y':
                self.save(page, original_text, new_text, applied,
                          show_diff=False, asynchronous=True)

            # choice must be 'N'
            break

        if self.opt.always and new_text != original_text:
            self.save(page, original_text, new_text, applied,
                      show_diff=False, asynchronous=False)

    def save(self, page, oldtext, newtext, applied, **kwargs) -> None:
        """Save the given page."""
        self.userPut(page, oldtext, newtext,
                     summary=self.generate_summary(applied),
                     ignore_save_related_errors=True, **kwargs)

    def user_confirm(self, question) -> bool:
        """Always return True due to our own input choice."""
        return True


def prepareRegexForMySQL(pattern: str) -> str:
    """Convert regex to MySQL syntax."""
    pattern = pattern.replace(r'\s', '[:space:]')
    pattern = pattern.replace(r'\d', '[:digit:]')
    pattern = pattern.replace(r'\w', '[:alnum:]')
    pattern = pattern.replace("'", '\\' + "'")
    return pattern


EXC_KEYS = {
    '-excepttitle': 'title',
    '-requiretitle:': 'require-title',
    '-excepttext': 'text-contains',
    '-exceptinside': 'inside',
    '-exceptinsidetag': 'inside-tags'
}
"""Dictionary to convert exceptions command line options to exceptions keys.

    .. versionadded:: 7.0
"""


def handle_exceptions(*args: str) -> Tuple[List[str], Dict[str, str]]:
    """Handle exceptions args to ignore pages which contain certain texts.

    .. versionadded:: 7.0
    """
    exceptions = {key: [] for key in EXC_KEYS.values()}
    local_args = []
    for argument in args:
        arg, _, value = argument.partition(':')
        if arg in EXC_KEYS:
            exceptions[EXC_KEYS[arg]].append(value)
        else:
            local_args.append(argument)
    return local_args, exceptions


def handle_sql(sql: str,
               replacements: List[Pattern],
               exceptions: List[Pattern]) -> Generator:
    """Handle default sql query.

    .. versionadded:: 7.0
    """
    if not sql:
        where_clause = 'WHERE ({})'.format(' OR '.join(
            "old_text RLIKE '{}'"
            .format(prepareRegexForMySQL(repl.old_regex.pattern))
            for repl in replacements))

        if exceptions:
            except_clause = 'AND NOT ({})'.format(' OR '.join(
                "old_text RLIKE '{}'"
                .format(prepareRegexForMySQL(exc.pattern))
                for exc in exceptions))
        else:
            except_clause = ''

        sql = """
SELECT page_namespace, page_title
FROM page
JOIN text ON (page_id = old_id)
{}
{}
LIMIT 200""".format(where_clause, except_clause)

    return pagegenerators.MySQLPageGenerator(sql)


def main(*args: str) -> None:
    """
    Process command line arguments and invoke bot.

    If args is an empty list, sys.argv is used.

    :param args: command line arguments
    """
    options = {}
    gen = None
    # summary message
    edit_summary = ''
    # Array which will collect commandline parameters.
    # First element is original text, second element is replacement text.
    preload = True  # preload pages
    # the dump's path, either absolute or relative, which will be used
    # if -xml flag is present
    xmlFilename = None
    xmlStart = None
    sql_query = None  # type: Optional[str]
    # Set the default regular expression flags
    flags = 0
    # Request manual replacements even if replacements are already defined
    manual_input = False

    # Read commandline parameters.
    genFactory = pagegenerators.GeneratorFactory(
        disabled_options=['mysqlquery'])
    local_args = pywikibot.handle_args(args)
    local_args = genFactory.handle_args(local_args)
    local_args, exceptions = handle_exceptions(*local_args)

    for arg in local_args:
        opt, _, value = arg.partition(':')
        if opt == '-xmlstart':
            xmlStart = value or pywikibot.input(
                'Please enter the dumped article to start with:')
        elif opt == '-xml':
            xmlFilename = value or i18n.input('pywikibot-enter-xml-filename')
        elif opt == '-mysqlquery':
            sql_query = value
        elif opt == '-sleep':
            options['sleep'] = float(value)
        elif opt == '-addcat':
            options['addcat'] = value
        elif opt == '-summary':
            edit_summary = value
        elif opt == '-automaticsummary':
            edit_summary = True
        elif opt == '-nopreload':
            preload = False

    if not textblob_imported:
        pywikibot.error('textblob module not available. Try "pip install textblob"?')
        return

    try:
        textblob.TextBlob("Test sentence").tags
    except textblob.exceptions.MissingCorpusError:
        pywikibot.error('corpora missing. Try "python -m textblob.download_corpora"?')
        return

    # The summary stored here won't be actually used but is only an example
    site = pywikibot.Site()
    single_summary = None

    if ((not edit_summary or edit_summary is True)
            and (single_summary)):
        if single_summary:
            pywikibot.output('The summary message for the command line '
                             'replacements will be something like: '
                             + single_summary)
        if edit_summary is not True:
            edit_summary = pywikibot.input(
                'Press Enter to use this automatic message, or enter a '
                'description of the\nchanges your bot will make:')
        else:
            edit_summary = ''

    if xmlFilename:
        gen = XmlDumpReplacePageGenerator(xmlFilename, xmlStart,
                                          replacements, exceptions, site)
    elif sql_query is not None:
        # Only -excepttext option is considered by the query. Other
        # exceptions are taken into account by the ReplaceRobot
        gen = handle_sql(sql_query, replacements, exceptions['text-contains'])

    gen = genFactory.getCombinedGenerator(gen, preload=preload)
    if pywikibot.bot.suggest_help(missing_generator=not gen):
        return

    bot = ReplaceRobot(gen, exceptions, site=site, summary=edit_summary,
                **options)
    site.login()
    bot.run()


if __name__ == '__main__':
    main()
