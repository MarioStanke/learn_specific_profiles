# functions for plotting and visualization

import cv2
import logging
import logomaker
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import plotly.express as px
import plotly.graph_objects as go

from .GeneLinkDraw import geneLinkDraw as gld
from . import Links
from . import SequenceRepresentation
#from .model import TrainingHistory, ProfileTracking
from .typecheck import typecheck_list

# set logging level for logomaker to avoid debug message clutter
logging.getLogger('logomaker').setLevel(logging.WARNING)

# try to find and use the font from this repo
try:
    _this_dir = str(os.path.dirname( os.path.abspath(__file__) ))
    _font_path = os.path.join(_this_dir, 'font', 'NugoSansLight-9YzoK.ttf')
    if not os.path.isfile(_font_path):
        _font_path = "/opt/conda/fonts/Ubuntu-M.ttf"
        logging.warning(f"[plotting] >>> Font file {_font_path} not found, falling back on {_font_path}")
except Exception as e:
    _font_path = "/opt/conda/fonts/Ubuntu-M.ttf"
    logging.warning(f"[plotting] >>> Error while searching for 'NugoSansLight-9YzoK.ttf': {e}. " \
                    +"Falling back on {_font_path}")


#def plotHistory(history: TrainingHistory):
def plotHistory(history):
    """ 
    Plot the training history as loss and accuracy curves 
    
    Parameters
        history (model.TrainingHistory): History object from trained model

    Returns
        fig, ax: matplotlib figure and axes objects
    """

    epochs = range(1, len(history.loss) + 1)

    fig, ax = plt.subplots(ncols = 2, figsize = (15, 6))
    ax[0].plot(epochs, history.loss, 'bo', label = 'Training loss')
    ax[0].set_title('Training loss')
    ax[0].legend()

    ax[1].plot(epochs, history.Rmax, 'bo', label = 'Rmax')
    ax[1].plot(epochs, history.Rmin, 'b+', label = 'Rmin')
    ax[1].plot(epochs, history.Smax, 'go', label = 'Smax')
    ax[1].plot(epochs, history.Smin, 'g+', label = 'Smin')
    ax[1].set_title('Training R and S')
    ax[1].legend()
    
    return fig, ax



def plotLogo(P: np.ndarray, alphabet: list[str], drop:list[str] = ['*'],
             idxarray = None, pNames = None, pScores = None, pLosses = None, max_print=5, ax=None):
    """
    Create logo(s) from profile(s)

    Parameters
        P (np.ndarray | tf.Tensor): tensor of shape (k, alphabet_size, U)
        alphabet (list of str): list of characters for the logo
        drop (list[str]): list of characters to drop from the alphabet. Default: ['*']
        idxarray (list of int): optional list of indices in P's third axis to plot only certain profiles
        pNames (list): optional list of names to assign for each profile in P (if None, index of profile will be 
                       displayed)
        pScores (list of float): optional list of scores for each profile in P
        pLosses (list of float): optional list of losses for each profile in P
        max_print (int): print up to this many logos
        ax (list of matplotlib axes): optional axes from a matplotlib Figure to plot on, one axes for each plot
    """

    if hasattr(P, 'numpy'): # catch tf.Tensor
        P = P.numpy()
    
    assert P.shape[1] == len(alphabet), \
        f"[plotting.plotLogo] alphabet size {len(alphabet)} does not match profile matrix shape {P.shape}"
    
    dfs = []
    for j in range(P.shape[2]): # U
        profile_matrix = P[:,:,j]
        df = pd.DataFrame(profile_matrix)
        df.columns = alphabet
        for dc in drop:
            if dc in df.columns:
                df = df.drop([dc], axis=1)
        
        dfs.append(df)

    for i in range(min(P.shape[2], max_print)):
        j = idxarray[i] if idxarray is not None else i
        profile_df = dfs[j]
        if ax is None:
            logo = logomaker.Logo(profile_df, vpad=.1, width=1)
        else:
            logo = logomaker.Logo(profile_df, vpad=.1, width=1, ax=ax[i])
            
        logo.style_xticks(anchor=0, spacing=1, rotation=45)
        logo.ax.set_ylabel('information (bits)')
        nametext = "Profile "+str(pNames[j]) if pNames is not None else f"Profile {j}"
        scoretext = (f" score={pScores[j]:.3f}") if pScores is not None else ""
        losstext = (f" loss={pLosses[j]:.3f}") if pLosses is not None else ""
        logo.ax.set_title(nametext + scoretext + losstext)



#def plotProfileLossHeatmap(Ptrack: ProfileTracking, figsize=(2*1920/100,2*1080/100), dpi=100):
def plotProfileLossHeatmap(Ptrack, figsize=(2*1920/100,2*1080/100), dpi=100):
    """ Plot a heatmap of each profile's loss over training epochs """
    losses = Ptrack.mean_losses #np.transpose(Ptrack.mean_losses)
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
    hm = ax.imshow(losses, interpolation = 'nearest', cmap="rainbow")
    plt.colorbar(hm)
    return fig, ax



#def plotBestProfileLossScatter(Ptrack: ProfileTracking):
def plotBestProfileLossScatter(Ptrack):
    """ Plot a scatter plot of the resp. best profile loss over training epochs """
    bestloss = list(np.min(Ptrack.mean_losses, axis=0))   # (U, n_epochs) -> (n_epochs,)
    bestidx = list(np.argmin(Ptrack.mean_losses, axis=0))
    fig = px.scatter(x = list(range(len(bestloss))),
                     y = bestloss, 
                     color = bestidx)
    return fig            



def drawGeneLinks(links: list[Links.Link | Links.MultiLink], 
                  genomes: list[SequenceRepresentation.Genome] = None, 
                  imname = None,
                  kmerSites: list[Links.Occurrence] = None, kmerCol = None, 
                  maskingSites: list[Links.Occurrence] = None, maskingCol = 'darkred',
                  onlyLinkedGenes = False,
                  font = None, **kwargs) -> Image.Image:
    """
    Draw an image with sequences as horizontal bars and links as connecting lines. Returns a PIL.Image object, which
    you should remember to close!

    Parameters:
        links (list of Links.Link or Links.MulitLink): links between sequence positions (occurrences)
        genomes (list of SequenceRepresentation.Genome): genomes to draw, optional. If None, only the sequences from the
                                                         links are drawn, otherwise all sequences in genomes are drawn
        imname (str): path to image file to write the image to, set to None for no writing
        kmerSites (list of Links.Occurrence): Optional list of tuples Links.Occurrences of initial kmer positions. 
                                              If given, these occurrences are drawn as small dots on the genes
        kmerCol (str or tuple of RGB values): Optional color used when drawing initial kmer positions. If None and
                                              kmerSites are given, color is determined automatically. Use gld.Palette
                                              class to get some color names
        maskingSites (list of Links.Occurrence): Optional list of tuples of Links.Occurrences of sites where DNA was
                                                 softmasked during training
        maskingCol (str or tuple of RGB values): Optional color used when drawing masking sites. Defaults to darkred. If
                                                 None and maskingSites are given, color is determined automatically. Use
                                                 gld.Palette class to get some color names
        onlyLinkedGenes (bool): If True, only draw genomes that have links
        font (str): optional, path to an alternative font file to use for text
        **kwargs: named arguments forwarded to gld.draw()
    """
    if font is None:
        font = _font_path

    for link in links:
        typecheck_list(link, ['Link', 'MultiLink'], die=True)

    # if genomes is none, extract all genomes from links. Otherwise, assert that all links and sites are in genomes
    if genomes is None:
        genomes = []
        speciesToGenomeIdx = {}
        for link in links:
            occs = link.occs if link.classname == 'Link' else [occ for loccs in link.occs for occ in loccs]
            for occ in occs:
                if occ.sequence.species not in speciesToGenomeIdx:
                    speciesToGenomeIdx[occ.sequence.species] = len(genomes)
                    genomes.append(SequenceRepresentation.Genome([occ.sequence]))
                elif occ.sequence not in genomes[speciesToGenomeIdx[occ.sequence.species]]:
                    genomes[speciesToGenomeIdx[occ.sequence.species]].addSequence(occ.sequence)
    else:
        occs = []
        for link in links:
            occs += link.occs if link.classname == 'Link' else [occ for loccs in link.occs for occ in loccs]
        if kmerSites is not None:
            occs.extend(kmerSites)
        if maskingSites is not None:
            occs.extend(maskingSites)
        for occ in occs:
            assert any([occ.sequence in g for g in genomes]), f"Sequence {occ.sequence.id} not found in genomes"

    drawGenes = []
    seqidToDrawGeneId = {}
    for genome in genomes:
        for sequence in genome:
            dg = gld.Gene(sequence.id, sequence.species, sequence.length, sequence.strand)
            for element in sequence.genomic_elements:
                start, end = element.getRelativePositions(sequence, from_rc = False) # Always drawing forward strand
                dg.addElement(element.type, start, end-1) # TODO: refactor gld to also use exclusive end positions!

            drawGenes.append(dg)
            assert sequence.id not in seqidToDrawGeneId, f"Duplicate sequence id {sequence.id}"
            seqidToDrawGeneId[sequence.id] = dg.id

    # create links to draw
    drawLinks = []
    for link in links:
        if link.classname == 'Link':
            lgenes = []
            lpos = []
            for occ in link:
                dgid = seqidToDrawGeneId[occ.sequence.id]
                lgenes.append(dgid)
                lpos.append(occ.position + (occ.sitelen//2)) # probably not noticable, but this is the site center
                
            drawLinks.append(gld.Link(lgenes, lpos))
        else:
            lgenes = []
            lpos = []
            for goccs in link.occs:
                # in MultiLink, all occs from a sub-list are on the same gene
                lgenes.append( seqidToDrawGeneId[goccs[0].sequence.id] )
                lpos.append( [occ.position + (occ.sitelen//2) for occ in goccs] )

            drawLinks.append(gld.Link(lgenes, lpos, compressed=True))

    # also create kmer-"Link" showing the position of initial kmers and/or masking-"Link" to see where masking happened
    def createAdditionalSites(sites: list[Links.Occurrence], col):
        drawgeneidToOccs = {}
        for occ in sites:
            dgid = seqidToDrawGeneId[occ.sequence.id]
            if dgid not in drawgeneidToOccs:
                drawgeneidToOccs[dgid] = []

            drawgeneidToOccs[dgid].append(occ.position + (occ.sitelen//2))

        if len(drawgeneidToOccs.keys()) >= 2:
            # no links possible if less than two genomes
            lgenes = []
            lpos = []
            for gid in drawgeneidToOccs.keys():
                lgenes.append(gid)
                lpos.append(drawgeneidToOccs[gid])

            drawLinks.append(gld.Link(lgenes, lpos, connect=False, compressed=True, color=col))
        else:
            logging.warning("[plotting.drawGeneLinks.createAdditionalSites] >>> Could not "+\
                            "create kmer sites or masking sites because less than 2 genes are involved")
        

    if kmerSites is not None:
        createAdditionalSites(kmerSites, kmerCol)
    if maskingSites is not None:
        createAdditionalSites(maskingSites, maskingCol)

    # if desired, only draw genes that have links
    if onlyLinkedGenes:
        linkedGenes = set()
        for link in drawLinks:
            linkedGenes.update(link.genes)
        drawGenes = [dg for dg in drawGenes if dg.id in linkedGenes]

    # avoid masking kwargs and set defaults here that can be overwritten in function call
    gw = 20 if 'genewidth' not in kwargs else kwargs['genewidth']
    kwargs.pop('genewidth') if 'genewidth' in kwargs else ()
    lw = 1  if 'linkwidth' not in kwargs else kwargs['linkwidth']
    kwargs.pop('linkwidth') if 'linkwidth' in kwargs else ()
    img, _ = gld.draw(drawGenes, drawLinks, font = font,
                      genewidth = gw, linkwidth = lw, #width = (1920*2), 
                      **kwargs)
    if imname:
        img.save(imname)

    return img



def makeVideo(path, video_name, fps = 1):
    """
    Given a directory containing images, create a video from these images and store it there

    Parameters
        path (str): path to the directory containing the images, video is stored here
        video_name (str): filename of the created video
        fps (int): frames per second, i.e. how many images are shown in the video per second
    """

    assert os.path.isdir(path), str(path)+" not found or is not a directory"

    images = sorted([img for img in os.listdir(path) if img.endswith("png")])
    frame = cv2.imread(os.path.join(path, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, _ = frame.shape

    video = cv2.VideoWriter(os.path.join(path, video_name), cv2.VideoWriter_fourcc(*'H264'), fps, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(path, image)))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()



# === Stuff for histograms ===

def ownHist_impl(ls, binSize=None, bins=None):
    """ Returns two lists, the first containing the bin indices and the second the item count per bin

    Arguments:
        ls: list of values to create histogram from
        binSize: optional bin size, if None it is determined to
                 yield at most 100 bins.
                 Is ignored if bins is not None
        bins:    optional list of bins to use, must have at least
                 one element and pairwise bin differences must all
                 be equal """
    if bins is None:
        maxls = np.nanmax(ls)
        minls = np.nanmin(ls)
        if binSize is None:    
            if maxls == minls:
                binSize = 1
            else:
                binSize = max(1, (maxls-minls)/100)

        lbin = math.floor(minls/binSize)
        rbin = math.floor(maxls/binSize)
        # intercept if binsize would lead to more than 100 bins
        if len(range(lbin,(rbin+1))) > 101:
            return ownHist_impl(ls, (maxls-minls)/100)

        bins = [b*binSize for b in range(lbin,(rbin+1))]
        vals = [0 for b in range(lbin,(rbin+1))]
    else:
        assert len(bins) >= 1, "bins must contain at least one element"
        lbin = int(bins[0])
        assert lbin == bins[0], "first bin must be convertible to int, but is "+str(bins[0])
        if len(bins) == 1:
            binSize = 1
            vals = [0]
        else:
            binSize = bins[1] - bins[0]
            assert all([math.isclose(bins[i]-bins[i-1], binSize) for i in range(1,len(bins))]), "bin sizes not equal: "\
                                                                  + str([bins[i]-bins[i-1] for i in range(1,len(bins))])
            vals = [0 for _ in bins]
        
    for x in ls:
        if not np.isnan(x):
            b = math.floor(x/binSize)
            i = b - lbin
            vals[i] += 1
        
    return bins, vals



def ownHist(ls, binSize=None, bins=None):
    """ Returns two lists, the first containing the bin indices and the second the item count per bin, based on all 
        items in `ls` that are not NaN. """
    return ownHist_impl(ls, binSize, bins)



def ownHistRel(ls, binSize=None, bins=None):
    """ Returns two lists, the first containing the bin indices and the second the relative frequencies per bin, based 
        on all items in `ls` that are not NaN."""
    bins, vals = ownHist(ls, binSize, bins)
    vals[:] = [v/len(ls) for v in vals]
    return bins, vals



def plotOwnHist(bins, vals, ylim=None, precision=None, ax=None, **kwargs):
    """ Plots a histogram based on the output of ownHist or ownHistRel. Returns the figure and axis objects from 
        plt.subplots.
        Arguments:
            bins: list of bin indices
            vals: list of item counts/rel. frequencies per bin
            ylim: optional tuple (a, b) where a < b, values outside this range are clipped
            precision: optional int, number of digits after the decimal point to use for bin labels
            ax: optional axes object to plot on, a new figure is created if None (default)
            kwargs: additional arguments passed to plt.bar """
    if len(bins) > 1:
        if precision is None:
            binsize = bins[1] - bins[0]
            p = 1
            while int((p*10)*binsize) != ((p*10)*binsize):
                p += 1
        else:
            p = precision

        bins = [np.format_float_positional(b, precision=p) for b in bins]

    if ylim is not None:
        assert type(ylim) == tuple, "ylim must be a tuple"
        assert ylim[0] < ylim[1], 'ylim must be a tuple (a, b) where a < b'
        vals = [max(ylim[0], min(ylim[1], y)) for y in vals]
        
    newfig = ax is None
    if newfig:
        fig, ax = plt.subplots(1, 1, figsize=(16,9))

    ax.bar(x = range(len(bins)),
           height = vals,
           tick_label = bins, **kwargs)
    
    if newfig:
        return fig, ax 
    else:
        return ax
    


def ownPlotlyHist(lists: dict[str, list], binSize=None, bins=None, **kwargs):
    """ Create a plotly histogram from a dictionary of lists. The keys of the dictionary are used as labels for the 
        traces. 
        Arguments:
            lists: dictionary of lists to create histograms from. Keys must be trace labels, values must be the lists.
            binSize: optional bin size, if None it is determined to yield at most 100 bins. 
                     Is ignored if bins is not None
            bins: optional list of bins to use, must have at least one element and pairwise bin differences must all be 
                  equal
            kwargs: additional arguments passed to go.Histogram"""
    
    allvals = list(set([v for label in lists for v in lists[label]]))
    allbins, _ = ownHist(allvals, binSize, bins)
    
    counts = {}
    for label in lists:
        _, c = ownHist(lists[label], bins=allbins)
        counts[label] = c
        
    fig = go.Figure()
    for label in counts:
        fig.add_trace(go.Bar(x=allbins, y=counts[label], name=label, **kwargs))
    
    return fig



# === Other stuff ===

def combinePlots(plots: list, 
                 rows: int, cols: int, 
                 out: str, 
                 space=0, labels=None, fontsize=36, fontpath=None,
                 **kwargs):
    """ 
    General function to concatenate plots (from disk) to a multi-plot image

    Parameters
        plots (list of str): paths to the images that should be concatenated
        rows (int): number of vertical plots
        cols (int): number of horizontal plots
        out (str): path to the output image
        space (int): space between two plots in pixels
        labels (list of str): optional list of plot labels (e.g. ['A)', 'B)'])
        fontsize (int): fontsize in pt
        fontpath (str): optional path to the desired font
        **kwargs: keyword arguments passed to PIL.Image.save() 
    """

    assert len(plots) <= rows*cols, "Too many plots"
    assert len(plots) > (rows-1)*cols, "Too few plots, reduce rows or columns"
    if labels is not None:
        if fontpath is not None:
            assert os.path.isfile(fontpath), "Font not found"
        else:
            #fontpath = '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'
            fontpath = _font_path

        if not os.path.isfile(fontpath):
            addLabels = False
            #print("[WARNING] >>> Ignoring labels, default fontpath not found: "+fontpath)
            logging.warning(f"[plotting.combinePlots] >>> Ignoring labels, default fontpath not found: {fontpath}")
        else:
            addLabels = True
            font = ImageFont.truetype(fontpath, fontsize)
    else:
        addLabels = False
        
    if addLabels:
        assert len(labels) == len(plots), "provide one label for each plot"
    
    # load images
    images = [Image.open(p) for p in plots]
    
    # arrange images row-wise
    arrange = np.empty([rows, cols, 3]) # last: [idx, w, h]
    arrange *= np.nan
    k = 0
    for i in range(rows):
        for j in range(cols):
            if k < len(plots):
                arrange[i,j,0] = k
                arrange[i,j,1] = images[k].width
                arrange[i,j,2] = images[k].height
                k += 1
        
    # get canvas dimensions    
    canvasWidth = int(max(np.nansum(arrange[:,:,1], axis=1)) + space*(cols-1))
    canvasHeight = int(max(np.nansum(arrange[:,:,2], axis=0)) + space*(rows-1))
    
    # assemble images
    im = Image.new('RGB', (canvasWidth, canvasHeight), color='white')
    if addLabels:
        draw = ImageDraw.Draw(im)
        
    y = 0
    for i in range(rows):
        x = 0
        y += 0 if i == 0 else int(np.nanmax(arrange[(i-1),:,2])) + space # add row heights
        for j in range(cols):
            x += 0 if j == 0 else int(np.nanmax(arrange[:,(j-1),1])) + space # add col widhts
            imgIdx = arrange[i,j,0]
            if not np.isnan(imgIdx):
                im.paste(images[int(imgIdx)], (x,y))
            if addLabels:
                draw.text((x,y), labels[int(imgIdx)], 'black', font)
    
    im.save(out, **kwargs)
    im.close()


def scatter(
    df: pd.DataFrame,
    columnX: str,
    columnY: str | list[str],
    byColumn: str | list[str] = None,
    lines: bool = False,
    title: str = None,
    label: str | list[str] = None,
    hover: str = None,
    xlab: str = None,
    ylab: str = None,
    # removeNAs: bool = True,
    fig: go.Figure = None,
    row=None,
    col=None,
):
    """Creates a scatter plot from two numeric columns in the DataFrame.
    Arguments:
        df: Pandas DataFrame
        columnX: str, column name of the x values
        columnY: str or list of str, column name(s) of the y values (multiple traces if list)
        byColumn: str or list of str, column name(s) of (a) categorical column(s) to color the data points, optional
        lines: bool, add lines between data points (default: False)
        title: str, plot title (if None, no title)
        label: str or list of str, legend label(s) (if None, column name(s) is/are used), ignored if byColumn is given
        hover: str, column name of a column to display on hover
        xlab: str, label of the x-axis (if None, column name is used)
        ylab: str, label of the y-axis (if None, column name is used)
        # removeNAs: bool, removes all NA elements in the relevant columns if True
        fig: Plotly GraphObjects Figure, optional, add traces to this figure instead of creating a new one
        row: int, optional, only relevant if `fig` is a Plotly Subplots Figure
        col: int, optional, only relevant if `fig` is a Plotly Subplots Figure"""
    
    # if removeNAs:
    #     df = (
    #         _removeNAs(df, [columnX, columnY])
    #         if byColumn is None
    #         else _removeNAs(df, [columnX, columnY, byColumn])
    #     )
    def _numConversion(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Try to convert `col` in df to numeric. Returns a copy of df. Raises an exception if conversion fails."""
        df = df.copy()
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            print(f"[ERROR] >>> Could not convert '{col}' to numeric:")
            raise e

        return df
    
    def _combineByColumns(df: pd.DataFrame, byColumns: list[str]) -> list[str]:
        """ Given a list of column names for categorical columns, returns a new column that contains as categories all
            combinations of the categories of the input columns. 
            Example:
            df with columns 'A' and 'B' and categories ['a', 'b'] and ['x', 'y'] respectively
            returns a new column 'A_B' with categories ['a_x', 'a_y', 'b_x', 'b_y'] """
        newcol = [] # add new category values here
        for _, row in df.iterrows():
            newcol.append(', '.join([f"{col} {row[col]}" for col in byColumns]))

        return newcol
    
    def getPlotlyMarkerSymbols(i: int = None) -> list[tuple]:
        from plotly.validators.scatter.marker import SymbolValidator
        """Returns a list of available marker symbols. List elements are 2-tuples with
        (symbol-id: int, symbol-name: str).
        Supply an optional argument `i` to directly get the i-th symbol tuple. Index overflow is allowed, i.e. if the
        symbol list has 54 elements, `i=56` would yield the third (i=2) element."""
        raw_symbols = SymbolValidator().values
        symbols = list()
        for j in range(0, len(raw_symbols), 3):
            symbol = raw_symbols[j]
            name = raw_symbols[j + 2]
            if "open" in name or "dot" in name:
                continue
            symbols.append((symbol, name))

        assert sorted(symbols) == sorted(
            set(symbols)
        ), f"[getPlotlyMarkerSymbols]: symbols not unique: {symbols}"
        return symbols if i is None else symbols[i % len(symbols)]


    mode = "lines+markers" if lines else "markers"

    if fig is None:
        fig = go.Figure()

    columns = columnY if type(columnY) is list else [columnY]
    labels = None if label is None else label if type(label) is list else [label]
    df = _numConversion(df, columnX)
    for i, columnY in enumerate(columns):
        df = _numConversion(df, columnY)
        if labels is None:
            label = columnY
        else:
            label = labels[i]

        hovertemplate = f"{columnX}: %{{x}}<br>{columnY}: %{{y}}<br>i: %{{customdata[0]}}"
        if hover is not None:
            hovertemplate += f"<br>{hover}: %{{customdata[1]}}"

        if byColumn is not None:
            if type(byColumn) is list:
                bc = '_'.join(byColumn)
                assert bc not in df.columns, \
                    f"[ERROR] >>> Column with name {bc} already exists in DataFrame"
                df[bc] = _combineByColumns(df, byColumn)
                byColumn = bc

            categories = sorted(set(df[byColumn]))
            for i, cat in enumerate(categories):
                if hover is not None:
                    customdata = np.stack((df[df[byColumn] == cat].index, df[df[byColumn] == cat][hover]), axis=-1)
                else:
                    customdata = np.stack((df[df[byColumn] == cat].index, df[df[byColumn] == cat][columnX]), axis=-1)

                fig.add_trace(
                    go.Scatter(
                        x=df[df[byColumn] == cat][columnX],
                        y=df[df[byColumn] == cat][columnY],
                        name=cat,
                        marker_symbol=getPlotlyMarkerSymbols(i)[0],
                        mode=mode,
                        hovertemplate=hovertemplate,
                        customdata=customdata,
                    ),
                    col=col,
                    row=row,
                )
        else:
            if hover is not None:
                customdata = np.stack((df.index, df[hover]), axis=-1)
            else:
                customdata = np.stack((df.index, df[columnX]), axis=-1) # columnX just so stacking works
            fig.add_trace(
                go.Scatter(x=df[columnX], y=df[columnY], name=label, mode=mode,
                           hovertemplate=hovertemplate, customdata=customdata),
                col=col,
                row=row,
            )

    fig.update_layout(
        {
            "yaxis": {"title": ylab if ylab is not None else columns[0]},
            "xaxis": {"title": xlab if xlab is not None else columnX},
        }
    )
    if title:
        fig.update_layout({"title": title})

    return fig