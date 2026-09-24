# Find the one canonical bibliography folder; preserve default TeX search paths.
use Cwd qw(abs_path);
use File::Basename qw(dirname);
my $reference_dir = abs_path(dirname(__FILE__) . '/references');
$ENV{'BIBINPUTS'} = $reference_dir . ':' . ($ENV{'BIBINPUTS'} // '') . ':';
$ENV{'BSTINPUTS'} = $reference_dir . ':' . ($ENV{'BSTINPUTS'} // '') . ':';
