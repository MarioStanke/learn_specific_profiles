#!/usr/bin/perl

#############################################################
# maxSubtreeSet
# gets the set of leaves of a tree that build the maximum subtree
# This version is able to handle a list of taxa with scores
# where the taxa with higher scores are preferred if they are equally distant
#
# usage: maxSubtreeSet inputtree [options]
#
# inputtree: a tree file in newick format
# output: subset of leaves in list format printed to STDOUT
#
#
# Lars Romoth, 13.07.2015
#############################################################

use strict;
use Getopt::Long;
use Bio::TreeIO;

# by Lars
# The script is able to calculate a maximum sub tree

my $usage .= "$0 input.newick [options]\n";
$usage .= "\n";
$usage .= "This script need a package form bioperl.\n";
$usage .= "It is enough to download and unzip the bioperl folder and export the path in the following way:\n";
$usage .= "\$ export PERL5LIB=/home/user/tools/BioPerl-1.6.1/\n";
$usage .= "\n";
$usage .= "It is necessary that all nodes have unique names, also the ancestral.\n";
$usage .= "If you use Bash, the following oneliner could be helpful:\n";
$usage .= "\$ perl -pe 'unless(\$i){\$i=0;}; while(\$_ =~ s/\\\)\\\:/\\\)Anc\$i\\\:/ ";
$usage .= "|| \$_ =~ s/\\\)\\\;/\\\)Anc\$i\\\;/){\$i++;}' input.tree >anc.tree\n";
$usage .= "\n";
$usage .= "options:\n";
$usage .= "--size=i\t\tSets the number of the subset the program should find. (default: int(\#leaves/10 + 2))\n";
$usage .= "--save=s\t\tSets the name of a file with a list of taxa you want to have in the output.\n";
$usage .= "        \t\tTaxa names should be followed by a numeric score. With equally distant taxa, those with higher";
$usage .=             "scores are preferred.\n";
$usage .= "--taxa=s\t\tSets only one taxa you want to have in the output.\n";
$usage .= "\n";

sub condenseTree {
    my ($tree, $lca, $nodeSet) = @_;
    foreach my $sN (@{$nodeSet}) {
        my $act = $sN;
        my $temp;
        while (!($act->id eq $lca->id)) {
            $temp = $act->ancestor;
            $tree->splice(-remove_id => $act->id);
            $act = $temp;
        }
    }
}

sub mostDistantNode{
    my ($tree, $sinkNode, $nodeSet, $score) = @_;
    my $max = 0;
    my $farestNode;
    foreach my $leaf (@{$nodeSet}) {
        my $distance = $tree->distance(-nodes => [$sinkNode, $leaf]);
        if ($max < $distance) {
            $max = $distance;
            $farestNode = $leaf;
        } elsif ($max == $distance) {
            if ($score->{$farestNode->id} < $score->{$leaf->id}) {
                $farestNode = $leaf;
            }
        }
    }
    return $farestNode;
}

if ($#ARGV < 0) {
    die "\nERROR: Unknown Call $ARGV\n\n$usage";
}
my $inputfilename = $ARGV[0];

my $input  = new Bio::TreeIO(-file   => $inputfilename,
                             -format => "newick");

my $tree = $input->next_tree;

my @allNodes = $tree->get_nodes;
my @leaves = $tree->get_leaf_nodes;

my $n = int($#leaves/10 + 2);
my $filenameSave;
my $taxa;

GetOptions('size=i' => \$n, 'save=s' => \$filenameSave, 'taxa=s' => \$taxa);

# $n has to be in the interval [2,#leaves]
if ($n < 2) { die "\nERROR: There is no maximum tree with just 1 node.\n\n$usage"; }
if ($n > $#leaves + 1) { die "\nERROR: There are not enough leaves in the tree.\n\n$usage"; }

# test, if every node has a unique taxon
my %taxaHash;
foreach (@allNodes) {
    if ($_->id eq "") { die "\nERROR: There is at least one node without taxon.\n\n$usage"; }
    if (defined $taxaHash{$_->id}) { die "\nERROR: The taxon ".$_->id." is not unique\n\n$usage"; }
    $taxaHash{$_->id} = 1;
}

# the set of identified nodes by this script
my @selection;

# a node that unified all chosen paths and leaves; starting point to find the next node
my $actualLca;

# initialization with predefined nodes that have to be chosen
my $emptyLineCount;
my %score; # hash to store scores for each taxon

if ($filenameSave) {
    my @saveTaxa;
    open(SAVE, "<$filenameSave") || die "\nCouldn't open $filenameSave.\n";
    while (<SAVE>) {
        chomp;
        if ($_ =~ m/^\s*$/) {
            $emptyLineCount++;
            next;
        }
        my @line = split(/\s+/, $_);
        my $taxon = shift @line;
        if (!defined $taxaHash{$taxon}) {
            die "\nERROR: The node $taxon from the list in $filenameSave is not in the tree.\n\n$usage";
        }
        push(@saveTaxa, $taxon);
        $score{$taxon} = shift @line; # store score for this taxon
    }
    if ($emptyLineCount >= 2) { "WARNING: There are several empty lines in $filenameSave.\n"; }
    if ($#saveTaxa < 0) { die "\nERROR: Could not store taxa that should be saved. Maybe $filenameSave is empty.\n"; }

    # get nodes to the input taxa list
    my @saveNodes;
    foreach (@saveTaxa) {
        push(@saveNodes, $tree->find_node(-id => $_));
    }

    # if there is only 1 node in the input list, we search directly for the farest node to add it and to complete the 
    #   initialization
    if ($#saveNodes == 0) { push(@saveNodes, mostDistantNode($tree, $saveNodes[0], \@leaves, \%score)); }

    # search for the actual lca of all nodes we already want to have in out output at this position
    $actualLca = $tree->get_lca(-nodes => \@saveNodes);

    # condense all edges and nodes we already used in the actual lca node
    condenseTree($tree, $actualLca, \@saveNodes);
    push(@selection, @saveNodes);
}

# initialization with single predefined node
if ($taxa) {
    # get node to the input taxa list
    my @saveNode;
    push(@saveNode, $tree->find_node(-id => $taxa));
    # there is only 1 node in the input list, so we search directly for the farest node to add it 
    #   and to complete the initialization
    push(@saveNode, mostDistantNode($tree, $saveNode[0], \@leaves));

    # search for the actual lca of all nodes we already want to have in out output at this position
    $actualLca = $tree->get_lca(-nodes => \@saveNode);

    # condense all edges and nodes we already used in the actual lca node
    condenseTree($tree, $actualLca, \@saveNode);
    push(@selection, @saveNode);
}

# initialization without predefined nodes
my @pair;
if (!$actualLca) {
    @leaves = $tree->get_leaf_nodes;
    my $maxDistNode1 = mostDistantNode($tree, $leaves[0], \@leaves);
    my $maxDistNode2 = mostDistantNode($tree, $maxDistNode1, \@leaves);
    push(@pair, $maxDistNode1);
    push(@pair, $maxDistNode2);
    push(@selection, @pair);
    $actualLca = $tree->get_lca(-nodes => \@pair);
}

# add in every step the farest additional node until there are $n ones in @selection
while ($n > $#selection+1) {
    condenseTree($tree, $actualLca, \@pair);
    @leaves = $tree->get_leaf_nodes;

    my $startnodeThis = mostDistantNode($tree, $actualLca, \@leaves);
    push(@selection, $startnodeThis);

    @pair=();
    push(@pair, $startnodeThis);
    push(@pair, $actualLca);
    $actualLca = $tree->get_lca(-nodes => [$startnodeThis, $actualLca]);
}

foreach (@selection) {
    print $_->id."\n";
}

# if the script should be extendet at this position, dont forget to condense the tree with the last @pair
# condenseTree($tree, $actualLca, \@pair);

